# symm/sym_engine/determining.py
import sympy as sp
from itertools import count
from .jet import symmetry_residual


def _monomials(vars_, D):
    x, t, u = vars_
    monos = []
    for du in range(D + 1):
        for dt in range(D + 1 - du):
            for dx in range(D + 1 - du - dt):
                monos.append(x**dx * t**dt * u**du)
    return monos


def ansatz(pde, D):
    x, t, u = pde.x, pde.t, pde.u
    monos = _monomials((x, t, u), D)
    ctr = count()
    coeffs = []
    parts = []
    for _label in ("tau", "xi", "phi"):
        cs = [sp.Symbol(f"c_{next(ctr)}") for _ in monos]
        coeffs.extend(cs)
        parts.append(sum(c * m for c, m in zip(cs, monos)))
    tau, xi, phi = parts
    return tau, xi, phi, coeffs


def determining_matrix(pde, D):
    tau, xi, phi, coeffs = ansatz(pde, D)
    residual = sp.expand(symmetry_residual(pde, tau, xi, phi))
    split_vars = pde.free_jet_syms() + [pde.x, pde.t, pde.u]
    poly = sp.Poly(residual, *split_vars)
    rows = []
    for _monom, c_expr in poly.terms():
        c_expr = sp.expand(c_expr)
        rows.append([c_expr.coeff(ci) for ci in coeffs])
    A = sp.Matrix(rows) if rows else sp.zeros(0, len(coeffs))
    return A, coeffs
