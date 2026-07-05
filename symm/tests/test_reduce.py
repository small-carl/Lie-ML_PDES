# symm/tests/test_reduce.py
import sympy as sp
from symm.sym_engine.jet import PDE
from symm.sym_engine.reduce import reduce_by

def _burgers():
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    return pde, nu

def test_travelling_wave_reduction_residual_zero():
    pde, nu = _burgers()
    c = sp.Symbol('c')
    gen = (sp.Integer(1), c, sp.Integer(0))   # d_t + c d_x
    red = reduce_by(pde, gen)
    assert red is not None
    assert sp.simplify(red.residual) == 0
    z = red.z
    U = red.U
    expected = nu * U(z).diff(z, 2) - (U(z) - c) * U(z).diff(z)
    assert sp.simplify(red.ode - expected) == 0 or sp.simplify(red.ode + expected) == 0

def test_steady_reduction_from_dt():
    pde, nu = _burgers()
    gen = (sp.Integer(1), sp.Integer(0), sp.Integer(0))   # d_t  -> c=0, z=x, u=U(x)
    red = reduce_by(pde, gen)
    assert red is not None
    assert sp.simplify(red.residual) == 0

def test_dx_alone_returns_none():
    pde, nu = _burgers()
    gen = (sp.Integer(0), sp.Integer(1), sp.Integer(0))   # d_x : tau=0, not a handled family
    assert reduce_by(pde, gen) is None

def test_projective_generator_returns_none():
    pde, nu = _burgers()
    t, x, u = pde.t, pde.x, pde.u
    gen = (t**2, t*x, x - t*u)   # projective: a=tau/t=t is NOT constant -> not a scaling family
    assert reduce_by(pde, gen) is None

def test_porous_scaling_reduction_residual_zero():
    import sympy as sp
    from symm.sym_engine.jet import PDE
    m = sp.Integer(2)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="porous", params={})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut - (m * pde.u**(m - 1) * ux**2 + pde.u**m * uxx)
    pde._init_onshell()
    t, x = pde.t, pde.x
    gen = (2*t, x, sp.Integer(0))           # porous m=2 scaling symmetry (a=2b-m e, b=1,e=0)
    red = reduce_by(pde, gen)
    assert red is not None
    assert sp.simplify(red.residual) == 0   # Barenblatt similarity reduction is exact
