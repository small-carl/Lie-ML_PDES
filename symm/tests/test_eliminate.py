# symm/tests/test_eliminate.py
import sympy as sp
from symm.sym_engine.jet import PDE
from symm.sym_engine.eliminate import discover

def _burgers():
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    return pde

def test_burgers_dimension_is_five_and_stabilizes():
    res = discover(_burgers(), Dmin=1, Dmax=4)
    assert res.dim == 5
    assert res.D == 2          # stabilizes at degree 2 (projective generator is degree 2)
    assert not res.stalled
    assert len(res.basis) == 5

def test_step_callback_fires_per_degree():
    seen = []
    discover(_burgers(), Dmin=1, Dmax=3, on_step=lambda i, s: seen.append(s["D"]))
    assert seen == [1, 2, 3]

def test_discovered_generators_are_real_symmetries():
    import sympy as sp
    from symm.sym_engine.jet import symmetry_residual
    from symm.sym_engine.determining import ansatz
    pde = _burgers()
    res = discover(pde, Dmin=1, Dmax=4)
    # rebuild the ansatz at the REPORTED degree and reconstruct each generator
    tau, xi, phi, coeffs = ansatz(pde, res.D)
    assert len(res.basis) == 5
    for m in res.basis:
        subs = {c: m.get(c, 0) for c in coeffs}
        T = tau.subs(subs); X = xi.subs(subs); P = phi.subs(subs)
        assert (T, X, P) != (0, 0, 0)
        # every discovered generator must satisfy the symmetry condition exactly
        assert sp.simplify(symmetry_residual(pde, T, X, P)) == 0

def test_single_degree_is_marked_stalled():
    pde = _burgers()
    res = discover(pde, Dmin=2, Dmax=2)
    assert res.stalled is True     # one degree cannot confirm stabilization
