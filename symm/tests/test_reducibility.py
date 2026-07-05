# symm/tests/test_reducibility.py
import sympy as sp
from symm.sym_engine.jet import PDE
from symm.sym_engine.reducibility import defect, reducibility_index

def _burgers():
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    return pde

def test_defect_zero_when_symmetry_exists():
    assert defect(_burgers(), D=2) < 1e-6

def test_index_formula_boundaries():
    assert reducibility_index(dim=5, delta=0.0) == 1.0          # dim>=d* -> 1
    assert reducibility_index(dim=1, delta=0.0) == 0.75         # 0.5+0.5*(1/2)
    assert reducibility_index(dim=0, delta=0.0) == 0.5          # barren, perfect near-symmetry
    assert reducibility_index(dim=0, delta=1.0) == 0.0          # structureless
