# symm/tests/test_m1_gate.py
import sympy as sp
from symm.sym_engine import Engine
from symm.sym_engine_cli import build_pde
from symm.sym_engine.crosscheck import numeric_nullity, agree
from symm.sym_engine.eliminate import discover

def test_m1_burgers_full_gate(tmp_path):
    pde = build_pde("burgers")
    rec = Engine(pde, workdir=str(tmp_path)).run(resume=False)
    assert rec.dim == 5 and rec.status == "CERTIFIED"
    assert any(r["residual_zero"] for r in rec.reductions)
    assert rec.signature["crosscheck_agrees"]

def test_m1_porous_scaling_family():
    pde = build_pde("porous")
    res = discover(pde, 1, 4)
    assert res.dim >= 2                      # at least translations + scaling
    assert agree(res.dim, numeric_nullity(pde, res.D))

def test_m1_negative_control_rejected():
    pde = build_pde("burgers")
    from symm.sym_engine.jet import symmetry_residual
    r = sp.simplify(symmetry_residual(pde, pde.u, sp.Integer(0), sp.Integer(0)))
    assert r != 0

def test_m1_resume_bit_identical(tmp_path):
    pde = build_pde("burgers")
    a = Engine(pde, workdir=str(tmp_path)).run(resume=False)
    b = Engine(build_pde("burgers"), workdir=str(tmp_path)).run(resume=True)
    assert (a.dim, a.status, round(a.index, 6)) == (b.dim, b.status, round(b.index, 6))
