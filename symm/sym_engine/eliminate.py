# symm/sym_engine/eliminate.py
import sympy as sp
from dataclasses import dataclass, field
from .determining import determining_matrix


@dataclass
class EliminationResult:
    dim: int
    D: int
    basis: list
    coeffs: list
    log: list = field(default_factory=list)
    stalled: bool = False


def _nullspace_maps(A, coeffs):
    vecs = A.nullspace()
    maps = []
    for v in vecs:
        maps.append({c: sp.nsimplify(v[i]) for i, c in enumerate(coeffs)})
    return maps


def discover(pde, Dmin=1, Dmax=4, on_step=None):
    log = []
    prev_dim = None
    prev_basis = None
    prev_coeffs = None
    result = None
    for step, D in enumerate(range(Dmin, Dmax + 1)):
        A, coeffs = determining_matrix(pde, D)
        dim = A.cols - A.rank()
        basis = _nullspace_maps(A, coeffs)
        log.append(f"degree D={D}: nullity={dim}")
        if on_step is not None:
            on_step(step, {"D": D, "dim": dim,
                           "coeffs": [sp.srepr(c) for c in coeffs],
                           "basis": [{sp.srepr(k): sp.srepr(v) for k, v in m.items()}
                                     for m in basis]})
        result = EliminationResult(dim=dim, D=D, basis=basis, coeffs=coeffs, log=list(log))
        if prev_dim is not None and dim == prev_dim:
            # stabilized: report the EARLIER degree AND its basis/coeffs (kept consistent)
            result.D = D - 1
            result.dim = prev_dim
            result.basis = prev_basis
            result.coeffs = prev_coeffs
            return result
        prev_dim = dim
        prev_basis = basis
        prev_coeffs = coeffs
    if result is not None:
        result.stalled = True
        log.append("WARNING: dimension did not stabilize within Dmax")
        result.log = list(log)
    return result
