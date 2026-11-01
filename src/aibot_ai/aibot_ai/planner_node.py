"""planner_node: natural-language tasks -> LLM skill plans -> skill server (Physical AI mode).

Subscribes  /aibot/chat/user        std_msgs/String  what the user asks (GUI chat panel, `chat` CLI)
            /aibot/mode, /aibot/perception/objects
Publishes   /aibot/chat/robot       std_msgs/String  replies, questions, plan summaries, results
            /aibot/planner/status   std_msgs/String  JSON {state, request, plan, step, report}
Services    /aibot/planner/cancel   std_srvs/Trigger (STOP, mode change: called by mode_manager_node)
Uses        /aibot/skills/execute   ExecuteSkill action (all motion), Ollama HTTP API (the LLM)

One request at a time, in a worker thread; a request while busy is refused with a hint to STOP
first. Outside AI mode requests are refused (no motion can happen anyway: the skill server is
gated too). The planning loop itself is aibot_ai.planning.agent (ROS-free, unit-tested).
"""
from __future__ import annotations

import contextlib
import json
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aibot_msgs.action import ExecuteSkill
from aibot_msgs.msg import DetectedObjectArray, PlaceTarget
from aibot_teleop.utils.mode_gate import ModeGate

from .backends.base import BackendError
from .backends.ollama import OllamaBackend
from .planning.agent import AgentConfig, PlannerAgent, Report, StepOutcome
from .planning.plan import Step, World, WorldItem


class RosSkillRunner:
    """SkillRunner on the real skill server and perception."""

    def __init__(self, node: "PlannerNode") -> None:
        self.node = node

    def world(self) -> World:
        arr = self.node.objects
        items, holding = [], {"left": "", "right": ""}
        for o in (arr.objects if arr else []):
            items.append(WorldItem(o.id, o.label, o.color, o.pose.position.x, o.pose.position.y,
                                   max(o.size.x, o.size.y), o.held_by, o.inside))
            if o.held_by in holding:
                holding[o.held_by] = o.id
        return World(items, holding)

    def refresh(self) -> None:
        self.run(Step("scan"), threading.Event())

    def goal(self, step: Step) -> ExecuteSkill.Goal:
        if step.skill == "place" and step.relation == "in":
            return ExecuteSkill.Goal(skill="place_in", object_id=step.target)
        if step.skill == "place":
            return ExecuteSkill.Goal(skill="place", object_id=step.object, place_target=PlaceTarget(
                relation=step.relation, reference_id=step.target, region=step.region))
        if step.skill == "look_at":
            return ExecuteSkill.Goal(skill="look_at", object_id=step.object, text=step.region)
        return ExecuteSkill.Goal(skill=step.skill, object_id=step.object)

    def run(self, step: Step, cancelled: threading.Event) -> StepOutcome:
        n = self.node
        n.publish_status("executing", step=step.text())
        if not n.skills.wait_for_server(timeout_sec=2.0):
            return StepOutcome(False, "INVALID", "skill server not running")
        done = threading.Event()
        fut = n.skills.send_goal_async(self.goal(step))
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(5.0) or not fut.result().accepted:
            return StepOutcome(False, "INVALID", "skill goal not accepted")
        handle = fut.result()
        res_fut = handle.get_result_async()
        got = threading.Event()
        res_fut.add_done_callback(lambda _f: got.set())
        cancel_sent = False
        while not got.wait(0.05):
            if cancelled.is_set() and not cancel_sent:
                handle.cancel_goal_async()          # the skill server also stops on STOP / mode change
                cancel_sent = True
        r = res_fut.result().result
        return StepOutcome(r.success, r.reason_code, r.message)


class PlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("planner_node")
        p = self.declare_parameter
        self.backend_name: str = p("llm_backend", "ollama").value
        url = p("llm.url", "http://localhost:11434").value
        self.model: str = p("llm.model", "granite4.2:3b").value
        if self.backend_name != "ollama":
            raise SystemExit(f"llm_backend '{self.backend_name}' not supported (only 'ollama'; the Anthropic "
                             "backend was cut, see README)")
        self.llm = OllamaBackend(self.model, url, p("llm.keep_alive", "30m").value, p("llm.num_ctx", 4096).value,
                                 p("llm.num_predict", 384).value, p("llm.temperature", 0.0).value,
                                 p("llm.timeout", 120.0).value, p("llm.think", False).value)
        self.warmup: bool = p("llm.warmup", True).value
        self.cb = ReentrantCallbackGroup()
        self.objects: DetectedObjectArray | None = None
        self.chat_pub = self.create_publisher(String, "/aibot/chat/robot", 20)
        self.status_pub = self.create_publisher(String, "/aibot/planner/status", 5)
        self.skills = ActionClient(self, ExecuteSkill, "/aibot/skills/execute", callback_group=self.cb)
        self.runner = RosSkillRunner(self)
        cfg = AgentConfig(p("max_replans", 2).value, p("repair_attempts", 1).value, p("history_turns", 3).value)
        self.agent = PlannerAgent(self.llm, self.runner, self.say, cfg, lambda m: self.get_logger().info(m))
        self.gate = ModeGate(self, "ai", True, self._on_mode_change)
        self._busy = threading.Lock()
        self.create_subscription(String, "/aibot/chat/user", self._on_user, 10, callback_group=self.cb)
        self.create_subscription(DetectedObjectArray, "/aibot/perception/objects", lambda m: setattr(self, "objects", m),
                                 5, callback_group=self.cb)
        self.create_service(Trigger, "/aibot/planner/cancel", self._srv_cancel, callback_group=self.cb)
        self.llm_ok, self.llm_error = False, ""
        threading.Thread(target=self._check_backend, daemon=True).start()
        self.publish_status("idle")

    # ------------------------------------------------------------------ backend
    def _check_backend(self) -> None:
        """Endpoint + model check at startup: a missing Ollama or model is reported, not assumed."""
        try:
            self.llm.check()
            self.llm_ok = True
            self.get_logger().info(f"LLM backend: ollama {self.model} at {self.llm.url}")
        except BackendError as exc:
            self.llm_error = str(exc)
            self.get_logger().error(f"LLM backend unavailable: {exc}")

    def _warm(self) -> None:
        """Load the model into memory when AI mode starts, so the first request does not pay the load."""
        if not (self.llm_ok and self.warmup):
            return
        t0 = time.monotonic()
        try:
            self.llm.generate([{"role": "user", "content": "Reply with {}"}], {"type": "object"})
            self.get_logger().info(f"LLM warm in {time.monotonic() - t0:.1f} s (load {self.llm.last_stats.get('load_s', 0):.1f} s)")
        except BackendError as exc:
            self.get_logger().warn(f"LLM warm-up failed: {exc}")

    # ------------------------------------------------------------------ chat
    def say(self, text: str) -> None:
        self.chat_pub.publish(String(data=text))
        self.get_logger().info(f"robot: {text}")

    def publish_status(self, state: str, **kw) -> None:
        self.status_pub.publish(String(data=json.dumps({"state": state, **kw})))

    def _on_mode_change(self, _old: str | None, new: str) -> None:
        if new != "ai":
            self.agent.cancel()
            return
        self.agent.stale = True           # mimic moved the head and arms: look at the table again first
        if self.llm_ok:
            threading.Thread(target=self._warm, daemon=True).start()

    def _srv_cancel(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self.agent.cancel()
        res.success, res.message = True, "plan cancelled" if self._busy.locked() else "no plan running"
        return res

    def _on_user(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        self.get_logger().info(f"user: {text}")
        if not self.gate.active:
            self.say(f"I'm in {self.gate.mode or 'an unknown'} mode. Switch to Physical AI to give me tasks.")
            return
        if not self.llm_ok:
            self._check_backend()
            if not self.llm_ok:
                self.say(f"My planner is offline: {self.llm_error}")
                return
        if not self._busy.acquire(blocking=False):
            self.say("I'm still working on the last task. Press STOP first if you want me to do something else.")
            return
        threading.Thread(target=self._work, args=(text,), daemon=True).start()

    def _work(self, text: str) -> None:
        try:
            self.publish_status("planning", request=text)
            rep: Report = self.agent.handle(text)
            self.get_logger().info(
                f"request done: kind={rep.kind} success={rep.success} llm_calls={rep.llm_calls} llm={rep.llm_s:.1f}s "
                f"repairs={rep.repairs} replans={rep.replans} first_motion="
                f"{'-' if rep.first_motion_s is None else f'{rep.first_motion_s:.1f}s'} total={rep.total_s:.1f}s "
                f"steps={rep.executed}")
            self.publish_status("idle", request=text, report=rep.as_dict())
        finally:
            self._busy.release()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PlannerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with contextlib.suppress(KeyboardInterrupt):
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
