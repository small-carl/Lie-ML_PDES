# symm/tests/test_report_interfaces.py
from symm.sym_engine.reducibility import ReducibilityRecord
from symm.sym_engine.report import render
from symm.sym_engine.interfaces import propose_reductions, review

def _rec():
    return ReducibilityRecord(
        name="burgers", dim=5, status="CERTIFIED", delta=0.0, index=1.0,
        params={"D": 2, "d_star": 2},
        algebra=[["Integer(1)", "Integer(0)", "Integer(0)"]],
        reductions=[{"gen": ["Integer(1)", "Symbol('c')", "Integer(0)"], "z": "Symbol('z')",
                     "ode": "x", "residual_zero": True}],
        signature={"dim": 5, "crosscheck_dim": 5, "crosscheck_agrees": True})

def test_render_contains_key_fields():
    txt = render(_rec())
    assert "burgers" in txt and "CERTIFIED" in txt and "index" in txt.lower()

def test_ranker_orders_by_simplicity():
    rec = _rec()
    rec.reductions.append({"gen": ["Add(Symbol('x'), Mul(Symbol('t'), Symbol('u')))",
                                   "Mul(Symbol('t'), Symbol('x'))", "Symbol('x')"],
                           "z": "z", "ode": "x", "residual_zero": True})
    ranked = propose_reductions(rec)
    def total(red):
        return sum(len(s) for s in red["gen"])
    assert total(ranked[0]) <= total(ranked[-1])     # simplest generator first
    assert total(ranked[0]) < total(ranked[1])       # the two differ, ordering is real

def test_review_no_secret_leak(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "SECRET123")
    out = review(_rec(), "steplog")
    assert out["deepseek_enabled"] is True
    assert "SECRET123" not in str(out)
    assert out["crosscheck_ok"] is True
    assert out["all_reductions_verified"] is True
