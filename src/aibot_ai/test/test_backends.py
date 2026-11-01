"""Backends against a local HTTP stub (no network): success, timeout, 429, malformed output, schema
rejection, circuit breaker, key handling, privacy rule."""
from __future__ import annotations

import ast
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from aibot_ai.backends.base import HEAD_CAMERA, BackendError, LLMBackend, PrivacyError, VLMBackend
from aibot_ai.backends.factory import build
from aibot_ai.backends.hybrid import CircuitBreaker, HybridBackend
from aibot_ai.backends.ollama import OllamaBackend
from aibot_ai.backends.openrouter import CloudError, OpenRouterBackend
from aibot_ai.backends.schema_check import schema_errors
from aibot_ai.backends.secrets import KEY_NAME, key_status, load_dotenv, parse_env

SCHEMA = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["a", "b"]}}, "required": ["kind"]}
FAKE_KEY = "test-not-a-real-key"


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *_a) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        srv = self.server
        srv.requests.append((body, dict(self.headers)))
        mode = srv.modes.pop(0) if len(srv.modes) > 1 else srv.modes[0]
        if isinstance(mode, int):
            self.send_response(mode)
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "stub says no"}}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if mode == "stall":
            time.sleep(1.0)
            return
        if mode == "keepalive_only":
            for _ in range(10):
                self.wfile.write(b": OPENROUTER PROCESSING\n\n")
                self.wfile.flush()
                time.sleep(0.1)
            return
        if mode == "stream_error":
            self.wfile.write(b'data: {"error": {"code": 502, "message": "provider down"}}\n\n')
            return
        text = {"ok": '{"kind": "a"}', "garbage": "not json", "bad_schema": '{"kind": "zzz"}'}[mode]
        for piece in (text[:4], text[4:]):
            chunk = {"choices": [{"delta": {"content": piece}}], "provider": "Stub"}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setenv(KEY_NAME, FAKE_KEY)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    srv.modes, srv.requests = ["ok"], []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def client(srv, **kw) -> OpenRouterBackend:
    kw = {"first_byte_timeout": 0.4, "total_timeout": 1.0, **kw}
    return OpenRouterBackend("m/x", url=f"http://127.0.0.1:{srv.server_address[1]}", **kw)


def test_success_and_request_format(stub):
    c = client(stub, reasoning_effort="minimal")
    assert c.generate([{"role": "user", "content": "hi"}], SCHEMA) == {"kind": "a"}
    body, headers = stub.requests[0]
    assert headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["provider"] == {"require_parameters": True, "sort": "latency"}
    assert body["reasoning"] == {"effort": "minimal"} and body["stream"] is True
    assert c.describe_last().startswith("cloud ")


def test_image_is_a_data_url_content_part(stub):
    client(stub).ask(b"\xff\xd8jpeg", "what?", SCHEMA, source=HEAD_CAMERA)
    content = stub.requests[0][0]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("mode,needle", [("stall", "timed out"), ("garbage", "non-JSON"), ("bad_schema", "schema"),
                                         ("stream_error", "stream error"), ("keepalive_only", "first byte")])
def test_failures_raise(stub, mode, needle):
    stub.modes = [mode]
    with pytest.raises(CloudError, match=needle):
        client(stub).generate([], SCHEMA)


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False), (401, False)])
def test_http_status_classification(stub, status, retryable):
    stub.modes = [status]
    with pytest.raises(CloudError) as e:
        client(stub).generate([], SCHEMA)
    assert e.value.status == status and e.value.retryable is retryable


def test_key_never_appears_in_errors_or_status(stub):
    stub.modes = [401]
    with pytest.raises(CloudError) as e:
        client(stub).generate([], SCHEMA)
    assert FAKE_KEY not in str(e.value) and FAKE_KEY not in key_status() and key_status().endswith("present")


class Fake(LLMBackend, VLMBackend):
    def __init__(self, kind, script):
        self.kind, self.name, self.script, self.calls = kind, kind, list(script), 0

    def _next(self):
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item

    def generate(self, messages, schema=None):
        return self._next()

    def ask(self, image_jpeg, prompt, schema=None, source=""):
        return self._next()


def hybrid(cloud, local, now, **kw):
    return HybridBackend([cloud, local], kw.get("failures", 2), 60.0, 0.0, clock=lambda: now[0], sleep=lambda _s: None)


def test_cloud_first_then_label():
    now = [0.0]
    h = hybrid(Fake("cloud", [{"k": 1}]), Fake("local", [{"k": 2}]), now)
    assert h.generate([]) == {"k": 1} and h.describe_last() == "cloud 0.0 s"


def test_falls_back_and_says_so():
    now = [0.0]
    h = hybrid(Fake("cloud", [CloudError("down")]), Fake("local", [{"k": 2}]), now)
    assert h.generate([]) == {"k": 2} and h.describe_last() == "local 0.0 s (cloud failed)"


def test_retries_once_on_429_only():
    now = [0.0]
    c = Fake("cloud", [CloudError("busy", 429, True), {"k": 1}])
    assert hybrid(c, Fake("local", [{"k": 2}]), now).generate([]) == {"k": 1} and c.calls == 2
    c2, l2 = Fake("cloud", [CloudError("bad", 400, False)]), Fake("local", [{"k": 2}])
    assert hybrid(c2, l2, now).generate([]) == {"k": 2} and c2.calls == 1


def test_breaker_skips_cloud_then_probes_and_recovers():
    now = [0.0]
    c, l = Fake("cloud", [CloudError("x")]), Fake("local", [{"k": 2}])
    h = hybrid(c, l, now)
    h.generate([]), h.generate([])
    assert c.calls == 2
    h.generate([])
    assert c.calls == 2 and h.describe_last() == "local 0.0 s (cloud down)"       # skipped, no delay
    now[0] = 61.0
    h.generate([])
    assert c.calls == 3                                                            # probe failed: open again
    h.generate([])
    assert c.calls == 3
    now[0] = 130.0
    c.script = [{"k": 1}]
    assert h.generate([]) == {"k": 1} and h.members[0].breaker.opened_at is None   # probe worked


def test_fallback_gets_the_short_prompt():
    seen = {}

    class Rec(Fake):
        def generate(self, messages, schema=None):
            seen[self.kind] = (messages, schema)
            return self._next()

    now = [0.0]
    h = hybrid(Rec("cloud", [CloudError("x")]), Rec("local", [{"k": 2}]), now)
    assert h.generate(["long"], {"a": 1}, local=(["short"], {"b": 2})) == {"k": 2}
    assert seen["cloud"] == (["long"], {"a": 1}) and seen["local"] == (["short"], {"b": 2})


def test_breaker_success_resets_the_count():
    b = CircuitBreaker(2, 60.0, clock=lambda: 0.0)
    b.failure(), b.success(), b.failure()
    assert b.allow()


def test_all_failing_raises():
    now = [0.0]
    with pytest.raises(BackendError):
        hybrid(Fake("cloud", [CloudError("x")]), Fake("local", [BackendError("y")]), now).generate([])


def test_no_key_degrades_to_local_with_one_notice(monkeypatch):
    monkeypatch.delenv(KEY_NAME, raising=False)
    notes = []
    h = build("hybrid", {"model": "m"}, {"model": "l"}, {}, notes.append)
    assert not h.members[0].backend.available and sum("local models only" in n for n in notes) == 1
    h.members[1].backend.generate = lambda m, s=None: {"k": 2}
    assert h.generate([]) == {"k": 2} and "cloud unavailable" in h.describe_last()
    with pytest.raises(SystemExit):
        build("anthropic", {"model": "m"}, {"model": "l"}, {})


# ---------------------------------------------------------------- privacy
@pytest.mark.parametrize("source", ["", "webcam", "unknown"])
def test_only_head_camera_images_reach_any_backend(stub, source):
    with pytest.raises(PrivacyError):
        client(stub).ask(b"jpeg", "q", SCHEMA, source=source)
    assert stub.requests == []                                     # nothing left the machine
    with pytest.raises(PrivacyError):
        OllamaBackend("m", url="http://127.0.0.1:9").ask(b"jpeg", "q", source=source)
    f = Fake("cloud", [{"k": 1}])
    with pytest.raises(PrivacyError):
        HybridBackend([f]).ask(b"jpeg", "q", source=source)
    assert f.calls == 0


def test_webcam_code_cannot_reach_models_or_the_network():
    """The webcam node and GUI import no backend or network module; the VLM node takes images only
    from head-camera topics."""
    root = Path(__file__).resolve().parents[2]
    for name in ("webcam_node.py", "camera_gui.py"):
        tree = ast.parse((root / "aibot_teleop" / "aibot_teleop" / name).read_text())
        mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not [m for m in mods if any(x in m for x in ("backends", "urllib", "http", "requests", "socket"))], name
    src = (root / "aibot_ai" / "aibot_ai" / "perception_node.py").read_text()
    topics = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)
              and n.value.startswith("/") and any(w in n.value for w in ("image", "webcam", "teleop"))]
    assert set(topics) <= {"/head_camera/image", "/head_camera/depth_image"}, topics


# ---------------------------------------------------------------- .env and schema subset
def test_env_parsing_tolerates_spaces_and_never_overrides(tmp_path, monkeypatch):
    assert parse_env("# c\n\nA= v1 \nexport B = 'v2'\nC\n") == {"A": "v1", "B": "v2"}
    monkeypatch.setenv("AIBOT_T1", "kept")
    monkeypatch.delenv("AIBOT_T2", raising=False)
    p = tmp_path / ".env"
    p.write_text("AIBOT_T1=new\nAIBOT_T2= fresh\n")
    assert load_dotenv(p)
    import os
    assert os.environ["AIBOT_T1"] == "kept" and os.environ["AIBOT_T2"] == "fresh"
    assert not load_dotenv(tmp_path / "nope")


def test_schema_subset():
    s = {"type": "object", "required": ["a"], "properties": {"a": {"type": "array", "maxItems": 1, "items": {"type": "string"}}}}
    assert schema_errors({"a": ["x"]}, s) == []
    assert schema_errors({}, s) and schema_errors({"a": [1]}, s) and schema_errors({"a": ["x", "y"]}, s) and schema_errors([], s)
