# symm/sym_engine/jet.py
import sympy as sp


def _jet_name(a, b):
    return "u_" + "x" * a + "t" * b if (a + b) > 0 else "u"


class PDE:
    def __init__(self, Delta, order, name, params=None):
        self.x, self.t, self.u = sp.symbols('x t u')
        self.order = int(order)
        self.name = name
        self.params = dict(params or {})
        self.jet = {(0, 0): self.u}
        for o in range(1, self.order + 2):
            for a in range(o + 1):
                b = o - a
                self.jet[(a, b)] = sp.Symbol(_jet_name(a, b))
        self.Delta = None if Delta is None else sp.sympify(Delta)
        self._onshell = {}
        if self.Delta is not None:
            self._init_onshell()

    @classmethod
    def scalar_1plus1(cls, Delta, order, name, params=None):
        return cls(Delta, order, name, params)

    def jet_sym(self, a, b):
        return self.jet[(a, b)]

    def leading_time_derivative(self):
        # the highest-order time derivative (0, b) that actually appears in Delta
        for b in range(self.order, 0, -1):
            if sp.diff(self.Delta, self.jet[(0, b)]) != 0:
                return self.jet[(0, b)]
        raise ValueError("Delta has no time derivative of order 1..order")

    def free_jet_syms(self):
        lead_sym = self.leading_time_derivative()
        lead_key = next(k for k, v in self.jet.items() if v == lead_sym)
        return [self.jet[k] for k in sorted(self.jet)
                if (k[0] + k[1]) >= 1 and k != lead_key]

    def _init_onshell(self):
        lead = self.leading_time_derivative()
        sol = sp.solve(sp.Eq(self.Delta, 0), lead, dict=True)
        if not sol:
            raise ValueError("Delta is not solvable for the leading time derivative")
        self._onshell = sol[0]

    def onshell(self):
        return dict(self._onshell)

    def _Dx(self, f):
        f = sp.sympify(f)
        res = sp.diff(f, self.x)
        for (a, b), s in self.jet.items():
            nxt = (a + 1, b)
            if nxt in self.jet:
                res += sp.diff(f, s) * self.jet[nxt]
        return res

    def _Dt(self, f):
        f = sp.sympify(f)
        res = sp.diff(f, self.t)
        for (a, b), s in self.jet.items():
            nxt = (a, b + 1)
            if nxt in self.jet:
                res += sp.diff(f, s) * self.jet[nxt]
        return res


def prolong(pde, tau, xi, phi):
    """Prolongation coefficients phi^{(a,b)} for 1 <= a+b <= pde.order."""
    tau, xi, phi = map(sp.sympify, (tau, xi, phi))
    P = {(0, 0): phi}
    for o in range(1, pde.order + 1):
        for a in range(o + 1):
            b = o - a
            if a >= 1:
                prev = P[(a - 1, b)]
                P[(a, b)] = (pde._Dx(prev)
                             - pde.jet[(a, b)] * pde._Dx(xi)
                             - pde.jet[(a - 1, b + 1)] * pde._Dx(tau))
            else:
                prev = P[(0, b - 1)]
                P[(0, b)] = (pde._Dt(prev)
                             - pde.jet[(1, b - 1)] * pde._Dt(xi)
                             - pde.jet[(0, b)] * pde._Dt(tau))
    return {k: v for k, v in P.items() if (k[0] + k[1]) >= 1}


def symmetry_residual(pde, tau, xi, phi):
    tau, xi, phi = map(sp.sympify, (tau, xi, phi))
    P = prolong(pde, tau, xi, phi)
    expr = (tau * sp.diff(pde.Delta, pde.t)
            + xi * sp.diff(pde.Delta, pde.x)
            + phi * sp.diff(pde.Delta, pde.u))
    for (a, b), coeff in P.items():
        expr += coeff * sp.diff(pde.Delta, pde.jet[(a, b)])
    return sp.expand(expr.subs(pde.onshell()))
