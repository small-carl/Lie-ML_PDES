# symm/sym_engine/crosscheck.py
import numpy as np
from .determining import determining_matrix


def numeric_matrix(pde, D):
    A, coeffs = determining_matrix(pde, D)
    A = A.subs(pde.params)
    M = np.array(A.evalf().tolist(), dtype=float) if A.rows else np.zeros((0, len(coeffs)))
    return M, coeffs


def _column_normalize(M):
    if M.size == 0:
        return M
    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0] = 1.0
    return M / norms


def numeric_nullity(pde, D, tol=1e-9):
    M, _ = numeric_matrix(pde, D)
    if M.size == 0:
        return M.shape[1]
    Mn = _column_normalize(M)
    s = np.linalg.svd(Mn, compute_uv=False)
    smax = s[0] if s.size else 0.0
    if smax == 0:
        return Mn.shape[1]
    return int(np.sum(s < tol * smax)) + (Mn.shape[1] - len(s))


def agree(exact_dim, num_dim):
    return int(exact_dim) == int(num_dim)
