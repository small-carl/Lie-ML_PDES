# symm/tests/test_engine.py
import sympy as sp
from symm.sym_engine import Engine, PDE

def _burgers():
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    return pde

def _barren():
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="barren", params={})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1)
    pde.Delta = ut - ux**2 + pde.x**3 * pde.u + pde.t**2 * ux   # confirmed barren: dim=0
    pde._init_onshell()
    return pde

def test_burgers_certified(tmp_path):
    rec = Engine(_burgers(), workdir=str(tmp_path)).run(resume=False)
    assert rec.dim == 5
    assert rec.status == "CERTIFIED"
    assert any(r["residual_zero"] for r in rec.reductions)
    assert rec.signature["crosscheck_agrees"]

def test_resume_is_identical(tmp_path):
    eng = Engine(_burgers(), workdir=str(tmp_path))
    full = eng.run(resume=False)
    resumed = Engine(_burgers(), workdir=str(tmp_path)).run(resume=True)
    assert resumed.dim == full.dim
    assert resumed.status == full.status
    assert round(resumed.index, 6) == round(full.index, 6)
    assert resumed.reductions == full.reductions
    assert resumed.params == full.params
    import os
    os.remove(os.path.join(str(tmp_path), "result.json"))
    recomputed = Engine(_burgers(), workdir=str(tmp_path)).run(resume=True)
    assert recomputed.dim == full.dim and recomputed.status == full.status
    assert recomputed.reductions == full.reductions

def test_barren_is_latent_only(tmp_path):
    rec = Engine(_barren(), workdir=str(tmp_path)).run(resume=False)
    assert rec.dim == 0
    assert rec.status == "LATENT-ONLY"
    assert rec.delta > 0.0
