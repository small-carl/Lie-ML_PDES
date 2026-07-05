# symm/tests/test_checkpoint.py
import sympy as sp
from symm.sym_engine.checkpoint import save_state, load_latest, encode, decode

def test_roundtrip_sympy(tmp_path):
    x = sp.Symbol('x')
    state = {"D": 2, "expr": x**2 + 1, "list": [sp.Integer(3), "ok"]}
    save_state(str(tmp_path), 0, state)
    save_state(str(tmp_path), 1, state)
    idx, loaded = load_latest(str(tmp_path))
    assert idx == 1
    assert loaded["D"] == 2
    assert sp.simplify(loaded["expr"] - (x**2 + 1)) == 0
    assert loaded["list"][1] == "ok"

def test_load_missing_returns_none(tmp_path):
    assert load_latest(str(tmp_path)) == (None, None)

def test_encode_decode_nested(tmp_path):
    x, y = sp.symbols('x y')
    obj = {"a": [x, {"b": y + 1}], "n": 5, "s": "hi"}
    back = decode(encode(obj))
    assert back["a"][0] == x
    assert sp.simplify(back["a"][1]["b"] - (y + 1)) == 0
    assert back["n"] == 5 and back["s"] == "hi"
