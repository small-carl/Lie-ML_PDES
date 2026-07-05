# symm/tests/test_jet.py
import sympy as sp
from symm.sym_engine.jet import PDE, symmetry_residual

def _burgers():
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    return pde, nu

def test_known_burgers_generators_have_zero_residual():
    pde, nu = _burgers()
    t, x, u = pde.t, pde.x, pde.u
    gens = [
        (sp.Integer(1), sp.Integer(0), sp.Integer(0)),          # d_t
        (sp.Integer(0), sp.Integer(1), sp.Integer(0)),          # d_x
        (sp.Integer(0), t, sp.Integer(1)),                      # Galilean
        (2*t, x, -u),                                           # scaling
        (t**2, t*x, x - t*u),                                   # projective
    ]
    for tau, xi, phi in gens:
        r = sp.simplify(symmetry_residual(pde, tau, xi, phi))
        assert r == 0

def test_non_symmetry_has_nonzero_residual():
    pde, nu = _burgers()
    r = sp.simplify(symmetry_residual(pde, pde.u, sp.Integer(0), sp.Integer(0)))  # tau=u: not a symmetry
    assert r != 0

def test_wave_equation_leading_derivative_is_u_tt():
    import sympy as sp
    from symm.sym_engine.jet import PDE
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="wave", params={})
    utt = pde.jet_sym(0, 2); uxx = pde.jet_sym(2, 0)
    pde.Delta = utt - uxx
    pde._init_onshell()
    assert pde.leading_time_derivative() == utt
    free = pde.free_jet_syms()
    assert utt not in free                      # the on-shell variable is excluded
    assert pde.jet_sym(0, 1) in free            # u_t is unconstrained -> free
    assert pde.jet_sym(1, 0) in free            # u_x is free

def test_burgers_free_syms_exclude_u_t():
    import sympy as sp
    from symm.sym_engine.jet import PDE
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    assert pde.leading_time_derivative() == ut
    assert ut not in pde.free_jet_syms()        # on-shell variable excluded
    assert ux in pde.free_jet_syms()
