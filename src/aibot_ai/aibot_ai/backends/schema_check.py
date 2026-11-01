"""Small JSON-Schema subset check of model output: type, enum, required, properties, items, maxItems.

Cloud providers enforce a schema with varying strictness, so the answer is checked here too; an
invalid one counts as a backend failure and the next backend is tried.
"""
from __future__ import annotations

TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}


def schema_errors(value: object, schema: dict, path: str = "$") -> list[str]:
    errs: list[str] = []
    t = schema.get("type")
    if t in TYPES and (not isinstance(value, TYPES[t]) or (t in ("integer", "number") and isinstance(value, bool))):
        return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in the allowed values")
    if isinstance(value, dict):
        errs += [f"{path}.{k}: missing" for k in schema.get("required", []) if k not in value]
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errs += schema_errors(value[k], sub, f"{path}.{k}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        for i, item in enumerate(value):
            errs += schema_errors(item, schema.get("items", {}), f"{path}[{i}]")
    return errs
