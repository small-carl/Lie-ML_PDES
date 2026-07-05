# symm/sym_engine/checkpoint.py
import os
import json
import glob
import sympy as sp


def encode(obj):
    if isinstance(obj, sp.Basic):
        return {"__srepr__": sp.srepr(obj)}
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return {"__repr__": repr(obj)}


def _from_srepr(s):
    return eval(s, {"__builtins__": {}}, dict(sp.__dict__))


def decode(obj):
    if isinstance(obj, dict):
        if "__srepr__" in obj:
            return _from_srepr(obj["__srepr__"])
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


def save_state(workdir, step_index, state):
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"checkpoint-{step_index:03d}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"step_index": step_index, "state": encode(state)}, f, indent=2)
    os.replace(tmp, path)
    return path


def load_latest(workdir):
    files = sorted(glob.glob(os.path.join(workdir, "checkpoint-*.json")))
    for path in reversed(files):
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
            return blob["step_index"], decode(blob["state"])
        except Exception:
            continue
    return None, None


def append_steplog(workdir, line):
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "steplog.md"), "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
