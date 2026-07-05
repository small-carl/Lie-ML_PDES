# symm/tests/test_crosscheck.py
import sympy as sp
from symm.sym_engine.jet import PDE
from symm.sym_engine.crosscheck import numeric_nullity, agree
from symm.sym_engine.eliminate import discover

def _burgers():
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    return pde

def test_numeric_nullity_matches_exact():
    pde = _burgers()
    exact = discover(pde, 1, 4).dim
    num = numeric_nullity(pde, D=2)
    assert num == exact == 5
    assert agree(exact, num)
