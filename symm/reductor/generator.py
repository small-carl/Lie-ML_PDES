"""Random scalar (1+1) evolution-PDE generator.

Two jobs: fuzz-test the symmetry engine on random inputs, and seed the
reducibility corpus that the learned reductor trains on. Deterministic given a seed.
"""
import numpy as np
import sympy as sp
from ..sym_engine.jet import PDE
from ..sym_engine.eliminate import discover

# (name, spatial_order, is_linear_term, builder)
MASTER = [
    ("u_xx",   2, True,  lambda p: p.jet[(2, 0)]),
    ("u_x",    1, True,  lambda p: p.jet[(1, 0)]),
    ("u",      0, True,  lambda p: p.u),
    ("uu_x",   1, False, lambda p: p.u * p.jet[(1, 0)]),
    ("u_x2",   1, False, lambda p: p.jet[(1, 0)] ** 2),
    ("uu_xx",  2, False, lambda p: p.u * p.jet[(2, 0)]),
    ("u2u_xx", 2, False, lambda p: p.u ** 2 * p.jet[(2, 0)]),
    ("u2",     0, False, lambda p: p.u ** 2),
    ("u3",     0, False, lambda p: p.u ** 3),
    ("u_xxx",  3, True,  lambda p: p.jet[(3, 0)]),
    ("xu",     0, True,  lambda p: p.x * p.u),
    ("xu_x",   1, True,  lambda p: p.x * p.jet[(1, 0)]),
]
COEFFS = [sp.Integer(1), sp.Integer(-1), sp.Integer(2), sp.Rational(1, 2), sp.Rational(-1, 2)]


def sample_pde(rng, p_include=0.4, max_order=2, name="gen"):
    """Sample u_t = sum_i c_i * term_i over the term dictionary (terms of order <= max_order)."""
    pool = [t for t in MASTER if t[1] <= max_order]
    while True:
        chosen = [t for t in pool if rng.random() < p_include]
        if chosen:
            break
    order = max(1, max(o for _, o, _, _ in chosen))
    pde = PDE.scalar_1plus1(Delta=None, order=order, name=name, params={})
    coeffs, rhs = {}, sp.Integer(0)
    for nm, o, lin, b in chosen:
        c = COEFFS[rng.integers(len(COEFFS))]
        coeffs[nm] = c
        rhs = rhs + c * b(pde)
    pde.Delta = pde.jet[(0, 1)] - rhs
    pde._init_onshell()
    pde._terms = [t[0] for t in chosen]
    pde._coeffs = coeffs
    return pde


def featurize(pde):
    """Presence (structure) features. Coefficient values were tested and HURT — dim is
    structure-determined, so magnitudes are noise for a distance classifier. `_coeffs` kept for reference."""
    present = set(getattr(pde, "_terms", []))
    return np.array([1.0 if name in present else 0.0 for name, *_ in MASTER])


def is_linear(pde):
    vars_ = [pde.u] + [pde.jet[k] for k in pde.jet if 1 <= sum(k) <= pde.order]
    try:
        return sp.Poly(sp.expand(pde.Delta), *vars_).total_degree() <= 1
    except sp.PolynomialError:
        return False


def label_pde(pde, Dmax=3):
    """Self-label via the engine: linearity (exact), finite dim, inf/fin class (stabilization)."""
    seq = []
    res = discover(pde, 1, Dmax, on_step=lambda i, s: seq.append(s["dim"]))
    lin = is_linear(pde)
    return {"linear": bool(lin), "dim": int(res.dim), "stalled": bool(res.stalled),
            "klass": "inf" if (lin or res.stalled) else "fin", "seq": list(seq),
            "terms": list(getattr(pde, "_terms", []))}


def build_corpus(n, seed=0, max_order=2, Dmax=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pde = sample_pde(rng, max_order=max_order, name=f"gen{i}")
        try:
            lab = label_pde(pde, Dmax=Dmax)
            lab["error"] = None
        except Exception as exc:  # record, do not crash the build
            lab = {"error": f"{type(exc).__name__}: {exc}", "terms": list(pde._terms)}
        rows.append({"feat": featurize(pde).tolist(), **lab})
    return rows
