# symm/sym_engine/reduce.py
import sympy as sp
from dataclasses import dataclass


@dataclass
class Reduction:
    gen: tuple
    z: sp.Symbol
    U: sp.Function
    ansatz: sp.Expr
    ode: sp.Expr
    residual: sp.Expr


def _is_const(e, vars_):
    return all(sp.diff(e, v) == 0 for v in vars_)


def reduce_by(pde, gen):
    x, t, u = pde.x, pde.t, pde.u
    tau, xi, phi = map(sp.sympify, gen)
    z = sp.Symbol('z', real=True)
    U = sp.Function('U')

    # --- translation family: tau, xi constant, phi == 0  -> travelling/standing wave ---
    if _is_const(tau, (x, t, u)) and _is_const(xi, (x, t, u)) and phi == 0 and tau != 0:
        c = sp.simplify(xi / tau)               # wave speed; z = x - c t, u = U(z)
        subs = {u: U(z)}
        # d^a_x d^b_t U(x - c t) = (-c)^b * U^{(a+b)}(z)
        for (a, b), s in pde.jet.items():
            if 1 <= a + b <= pde.order:
                subs[s] = (-c)**b * sp.diff(U(z), z, a + b)
        E = sp.expand(pde.Delta.subs(subs))
        remaining = [v for v in (x, t) if E.has(v)]
        residual = sp.simplify(E) if remaining else sp.Integer(0)
        return Reduction(gen=gen, z=z, U=U, ansatz=U(z), ode=sp.simplify(E), residual=residual)

    # --- scaling family: tau = a t, xi = b x, phi = e u ---
    if tau != 0:
        a = sp.simplify(tau / t)
        if _is_const(a, (x, t, u)) and a != 0 and sp.simplify(tau - a * t) == 0:
            b = sp.simplify(xi / x) if xi != 0 else sp.Integer(0)
            e = sp.simplify(phi / u) if phi != 0 else sp.Integer(0)
            if (_is_const(b, (x, t, u)) and _is_const(e, (x, t, u))
                    and sp.simplify(xi - b * x) == 0 and sp.simplify(phi - e * u) == 0):
                beta = sp.simplify(b / a)
                alpha = sp.simplify(e / a)
                u_of = t**alpha * U(x * t**(-beta))     # u = t^alpha U(z), z = x t^-beta
                subs = {u: u_of}
                for (aa, bb), s in pde.jet.items():
                    if 1 <= aa + bb <= pde.order:
                        d = u_of
                        for _ in range(aa):
                            d = sp.diff(d, x)
                        for _ in range(bb):
                            d = sp.diff(d, t)
                        subs[s] = d
                E = sp.expand(pde.Delta.subs(subs))
                # invert z = x t^-beta  (x = z t^beta) and factor the lowest power of t
                Ez = sp.powsimp(sp.expand(E.subs(x, z * t**beta)), force=True)
                Ez = sp.expand(sp.simplify(Ez))
                _terms = sp.Add.make_args(Ez)
                if _terms:
                    def _xt_factor(term):
                        fac = sp.Integer(1)
                        for f in sp.Mul.make_args(term):
                            if not f.has(z) and not f.has(U):
                                fac *= f
                        return fac
                    _common = _xt_factor(_terms[0])
                    cand = sp.simplify(Ez / _common) if _common != 0 else Ez
                else:
                    cand = Ez
                remaining = [v for v in (x, t) if cand.has(v)]
                residual = sp.Integer(0) if not remaining else sp.simplify(cand)
                return Reduction(gen=gen, z=z, U=U, ansatz=u_of, ode=cand, residual=residual)

    return None
