# symm/sym_engine/reducibility.py
import numpy as np
from dataclasses import dataclass, field
from .crosscheck import numeric_matrix, _column_normalize


@dataclass
class ReducibilityRecord:
    name: str
    dim: int
    status: str
    delta: float
    index: float
    params: dict
    algebra: list = field(default_factory=list)
    reductions: list = field(default_factory=list)
    signature: dict = field(default_factory=dict)


def defect(pde, D):
    M, _ = numeric_matrix(pde, D)
    if M.size == 0:
        return 0.0
    Mn = _column_normalize(M)
    s = np.linalg.svd(Mn, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0.0
    smin = s[-1] if Mn.shape[0] >= Mn.shape[1] else 0.0
    return float(smin / s[0])


def reducibility_index(dim, delta, d_star=2):
    if dim >= 1:
        return 0.5 + 0.5 * min(dim / d_star, 1.0)
    return 0.5 * (1.0 - delta)


def params_tuple(D, onshell_rule, d_star, param_values):
    return {"D": D, "onshell_rule": onshell_rule, "W_V": "column-normalized",
            "W_E": "identity", "d_star": d_star,
            "param_values": {str(k): float(v) for k, v in param_values.items()}}
