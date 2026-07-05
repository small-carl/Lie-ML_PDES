# symm/tests/test_algebra.py
import sympy as sp
from symm.sym_engine.jet import PDE
from symm.sym_engine.eliminate import discover
from symm.sym_engine.algebra import generators, bracket, structure_constants, algebra_type

def _burgers():
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    return pde

def test_five_generators_recovered():
    pde = _burgers()
    gens = generators(pde, discover(pde, 1, 4))
    assert len(gens) == 5

def test_translation_generators_commute():
    pde = _burgers()
    t, x, u = pde.t, pde.x, pde.u
    dt = (sp.Integer(1), sp.Integer(0), sp.Integer(0))
    dx = (sp.Integer(0), sp.Integer(1), sp.Integer(0))
    br = tuple(sp.simplify(c) for c in bracket(pde, dt, dx))
    assert br == (0, 0, 0)

def test_structure_constants_and_type_compute():
    pde = _burgers()
    gens = generators(pde, discover(pde, 1, 4))
    sc = structure_constants(pde, gens)
    info = algebra_type(sc)
    assert info["dim"] == 5
    # the Burgers point algebra contains an sl(2,R) subalgebra, hence is NON-solvable
    assert info["solvable"] is False
