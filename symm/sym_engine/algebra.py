# symm/sym_engine/algebra.py
import sympy as sp
from .determining import ansatz


def generators(pde, result):
    tau, xi, phi, coeffs = ansatz(pde, result.D)
    gens = []
    for m in result.basis:
        subs = {c: m.get(c, 0) for c in coeffs}
        g = (sp.expand(tau.subs(subs)), sp.expand(xi.subs(subs)), sp.expand(phi.subs(subs)))
        if any(comp != 0 for comp in g):
            gens.append(g)
    return gens


def bracket(pde, g1, g2):
    var = (pde.t, pde.x, pde.u)   # ordering matches generator tuple (tau, xi, phi)

    def apply(comps, f):
        return sum(comps[i] * sp.diff(f, var[i]) for i in range(3))

    return tuple(sp.expand(apply(g1, g2[i]) - apply(g2, g1[i])) for i in range(3))


def _coordinate_vector(pde, g, basis):
    x, t, u = pde.x, pde.t, pde.u
    syms = sp.symbols(f"a0:{len(basis)}")
    combo = [sum(syms[j] * basis[j][i] for j in range(len(basis))) for i in range(3)]
    eqs = []
    for i in range(3):
        diff = sp.expand(combo[i] - g[i])
        poly = sp.Poly(diff, x, t, u)
        eqs.extend(poly.coeffs())
    sol = sp.linsolve(eqs, syms)
    if not sol:
        raise ValueError("algebra not closed in basis")
    return list(next(iter(sol)))


def structure_constants(pde, gens):
    n = len(gens)
    C = [[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            br = bracket(pde, gens[i], gens[j])
            coords = _coordinate_vector(pde, br, gens)
            for k in range(n):
                C[k][i][j] = sp.nsimplify(coords[k])
    return sp.Array(C)


def _bracket_coords(C, n, a, b):
    out = []
    for k in range(n):
        s = sp.Integer(0)
        for i in range(n):
            for j in range(n):
                s += a[i] * b[j] * C[k, i, j]
        out.append(sp.expand(s))
    return out


def _span_basis(vectors, n):
    if not vectors:
        return []
    M = sp.Matrix.hstack(*[sp.Matrix(n, 1, list(v)) for v in vectors])
    return [list(col) for col in M.columnspace()]


def _series_dims(C, n, lower_central):
    full = [[sp.Integer(1) if i == j else sp.Integer(0) for i in range(n)] for j in range(n)]
    current = full
    dims = [n]
    while True:
        left = full if lower_central else current
        brackets = [_bracket_coords(C, n, a, b) for a in left for b in current]
        nb = _span_basis(brackets, n)
        d = len(nb)
        if d == 0:
            dims.append(0)
            break
        if d == dims[-1]:
            dims.append(d)
            break
        dims.append(d)
        current = nb
    return dims


def algebra_type(struct):
    C = struct
    n = C.shape[0]
    abelian = all(C[k, i, j] == 0 for k in range(n) for i in range(n) for j in range(n))
    derived = _series_dims(C, n, lower_central=False)
    lower = _series_dims(C, n, lower_central=True)
    return {"dim": n, "abelian": abelian, "nilpotent": lower[-1] == 0,
            "solvable": derived[-1] == 0, "derived_dims": derived}
