# Feasibility demo for: Semi-Analytical + ML methods for PDEs via Lie symmetry.
# Pure sympy/numpy/scipy. Demonstrates the CORE reduction step end-to-end:
#   (1) a general infinitesimal-prolongation engine for scalar (1+1) PDEs,
#   (2) verification that known Lie generators satisfy the invariance condition,
#   (3) symmetry reduction PDE -> ODE (travelling-wave & scaling), verified by substitution,
#   (4) exact closed-form solution of the reduced ODE (verified), and
#   (5) a semi-analytical series + a tiny "ML-style" convergence-control optimisation.
import sympy as sp
import numpy as np
from scipy.optimize import minimize_scalar

sp.init_printing()
t, x, nu, c, A = sp.symbols('t x nu c A', real=True)

# ---- jet coordinates (independent symbols on the jet space) ----
u   = sp.Symbol('u')
ut  = sp.Symbol('u_t');   ux  = sp.Symbol('u_x')
utt = sp.Symbol('u_tt');  utx = sp.Symbol('u_tx');  uxx = sp.Symbol('u_xx')
uttt= sp.Symbol('u_ttt'); uttx= sp.Symbol('u_ttx'); utxx= sp.Symbol('u_txx'); uxxx= sp.Symbol('u_xxx')

# total-derivative operators on the jet space (truncated to needed order)
jx = {u:ux, ut:utx, ux:uxx, utt:uttx, utx:utxx, uxx:uxxx}
jt = {u:ut, ut:utt, ux:utx, utt:uttt, utx:uttx, uxx:utxx}
def Dx(f):
    f = sp.sympify(f)
    return sp.diff(f, x) + sum(sp.diff(f, k)*v for k, v in jx.items())
def Dt(f):
    f = sp.sympify(f)
    return sp.diff(f, t) + sum(sp.diff(f, k)*v for k, v in jt.items())

def prolong2(tau, xi, phi):
    """Return dict of 2nd-order prolongation coefficients for X = tau d_t + xi d_x + phi d_u."""
    phit = Dt(phi) - ut*Dt(tau) - ux*Dt(xi)
    phix = Dx(phi) - ut*Dx(tau) - ux*Dx(xi)
    phitt = Dt(phit) - utt*Dt(tau) - utx*Dt(xi)
    phitx = Dx(phit) - utt*Dx(tau) - utx*Dx(xi)
    phixx = Dx(phix) - utx*Dx(tau) - uxx*Dx(xi)
    return dict(t=phit, x=phix, tt=phitt, tx=phitx, xx=phixx)

def invariance_residual(Delta, tau, xi, phi, onshell):
    """pr^(2)X(Delta) reduced on the solution manifold (onshell: dict jetvar->expr)."""
    P = prolong2(tau, xi, phi)
    prX = (tau*sp.diff(Delta, t) + xi*sp.diff(Delta, x) + phi*sp.diff(Delta, u)
           + P['t']*sp.diff(Delta, ut) + P['x']*sp.diff(Delta, ux)
           + P['tt']*sp.diff(Delta, utt) + P['tx']*sp.diff(Delta, utx)
           + P['xx']*sp.diff(Delta, uxx))
    return sp.simplify(sp.expand(prX.subs(onshell)))

print("="*70)
print("BENCHMARK 1 : viscous Burgers   u_t + u u_x = nu u_xx")
print("="*70)
Delta_b = ut + u*ux - nu*uxx           # = 0 on solutions
onshell_b = {ut: nu*uxx - u*ux}        # solve Delta=0 for u_t
gens = {
    "X1 = d_t                  (time translation)" : (1, 0, 0),
    "X2 = d_x                  (space translation)": (0, 1, 0),
    "X3 = t d_x + d_u          (Galilean boost)"   : (0, t, 1),
    "X4 = 2t d_t + x d_x - u d_u (scaling)"        : (2*t, x, -u),
    "X5 = t^2 d_t + t x d_x + (x - t u) d_u (projective)": (t**2, t*x, x - t*u),
}
all_ok = True
for name, (tau, xi, phi) in gens.items():
    r = invariance_residual(Delta_b, tau, xi, phi, onshell_b)
    ok = (r == 0)
    all_ok &= ok
    print(f"  invariance residual for {name:52s} = {r}   [{'OK' if ok else 'FAIL'}]")
print(f"  --> all five generators verified: {all_ok}")

# ---- travelling-wave reduction via X = d_t + c d_x  (invariant z = x - c t) ----
print("\n  Travelling-wave reduction (similarity var z = x - c t, u = U(z)):")
z = sp.Symbol('z', real=True)
U = sp.Function('U')
# substitute u_t=-c U', u_x=U', u_xx=U''
red = Delta_b.subs({ut: -c*sp.Derivative(U(z), z), ux: sp.Derivative(U(z), z),
                    uxx: sp.Derivative(U(z), z, 2), u: U(z)})
red = sp.simplify(red)
print(f"    reduced 2nd-order ODE:  {sp.Eq(red, 0)}")
# integrate once -> first-order ODE  nu U' = U^2/2 - c U - A.
# With A=(c^2-k^2)/2 this factors as  nu U' = (1/2)(U-c-k)(U-c+k)  -> separable -> tanh.
k = sp.Symbol('k', positive=True)
rhs_fac = sp.Rational(1, 2)*(U(z) - c - k)*(U(z) - c + k)
print(f"    first integral (A=(c^2-k^2)/2):  nu U' = {sp.expand(rhs_fac)}")
print(f"                                     = (1/2)(U-c-k)(U-c+k)   [separable]")
# explicit kink: U = c - k*tanh(k z/(2 nu)) solves -cU'+UU'-nuU''=0 ; verify by substitution
Ukink = c - k*sp.tanh(k*z/(2*nu))
chk = sp.simplify((-c*sp.diff(Ukink, z) + Ukink*sp.diff(Ukink, z) - nu*sp.diff(Ukink, z, 2)))
print(f"    verify tanh kink U=c-k*tanh(k z/2nu): residual = {chk}  [{'OK' if chk==0 else 'FAIL'}]")

print("\n" + "="*70)
print("BENCHMARK 2 : nonlinear/porous-medium heat   u_t = (u^m u_x)_x")
print("="*70)
m = sp.Symbol('m', positive=True)
Delta_p = ut - (m*u**(m-1)*ux**2 + u**m*uxx)     # = u_t - d_x(u^m u_x)
onshell_p = {ut: m*u**(m-1)*ux**2 + u**m*uxx}
# scaling generator: u_t=(u^m u_x)_x is invariant under  X = a t d_t + b x d_x + e u d_u
# with the relation  -a + 2b - m e = 0 (dimensional balance). Take b=1 => a = 2 - m e ... verify a family.
a, b, e = sp.symbols('a b e', real=True)
res_scaling = invariance_residual(Delta_p, a*t, b*x, e*u, onshell_p)
print(f"  invariance residual (general scaling a t d_t + b x d_x + e u d_u):")
print(f"    {sp.factor(res_scaling)} = 0")
res_sub = sp.simplify(res_scaling.subs(a, 2*b - m*e))
print(f"  --> impose  a = 2b - m e :  residual becomes  {res_sub}  "
      f"[{'symmetry confirmed' if res_sub == 0 else 'FAIL'}]")
# pick the mass-conserving self-similar generator: b=1, choose e so reduction is clean
# classical source (Barenblatt) reduction: u = t^(-alpha) F(eta), eta = x t^(-beta)
alpha, beta = sp.symbols('alpha beta', positive=True)
F = sp.Function('F')
eta = sp.Symbol('eta', real=True)
print("\n  Self-similar ansatz  u = t^(-alpha) F(eta),  eta = x t^(-beta):")
print("    mass conservation in 1-D fixes alpha = beta, and scaling fixes beta = 1/(m+2).")
print("    -> reduced ODE:  (F^m F')' + beta(eta F)' = 0   (Barenblatt ODE).")
# verify the Barenblatt closed form solves the reduced ODE for the source solution
# F(eta) = ( C - (beta*m/(2(m+1))) eta^2 )_+^(1/m)
C = sp.Symbol('C', positive=True)
print(f"    Barenblatt profile F = (C - (m*beta/2) eta^2)^(1/m), beta = 1/(m+2)")
print(f"    verify first integral  F^m F' + beta*eta*F = 0  at concrete m:")
for mv in (1, 2, 3):
    bv = sp.Integer(1)/(mv+2)
    Fv = (C - (mv*bv/2)*eta**2)**(sp.Integer(1)/mv)
    fi = sp.simplify(Fv**mv*sp.diff(Fv, eta) + bv*eta*Fv)
    print(f"      m={mv}: residual = {fi}   [{'OK' if fi == 0 else 'FAIL'}]")

print("\n" + "="*70)
print("STEP 3 : semi-analytical series + ML-style convergence-control on the")
print("         Burgers travelling-wave ODE   nu U'' = (U - c) U'  (1st integral A=0)")
print("="*70)
# Exact reference kink with chosen params
nu0, c0, k0 = 0.5, 1.0, 1.0
def U_exact(zz): return c0 - k0*np.tanh(k0*zz/(2*nu0))
# Semi-analytical power series U = sum a_n z^n solved by an explicit coefficient recurrence
# from the reduced ODE  nu U'' - (U-c)U' = 0, with ICs U(0)=c, U'(0)=-k^2/(2nu).
N = 8
zc = sp.symbols('z')
a = list(sp.symbols(f'a0:{N+1}'))
Up = sum(a[n]*zc**n for n in range(N+1))
ode_expr = sp.expand(nu0*sp.diff(Up, zc, 2) - (Up - c0)*sp.diff(Up, zc))
ode_expr = ode_expr.subs({a[0]: c0, a[1]: -k0**2/(2*nu0)})
unknowns = a[2:]
eqs = [sp.Eq(ode_expr.coeff(zc, j), 0) for j in range(0, N - 1)]   # determines a2..aN
coef = sp.solve(eqs, unknowns, dict=True)[0]
coef[a[0]] = c0; coef[a[1]] = -k0**2/(2*nu0)
series_poly = sum(coef[a[n]]*zc**n for n in range(N+1))
print(f"  {N}-term semi-analytical series solution:\n    U(z) ~ {sp.nsimplify(series_poly, rational=False)}")
ser_poly = sp.lambdify(zc, series_poly, 'numpy')
grid = np.linspace(-1.0, 1.0, 41)
err_series = np.max(np.abs(ser_poly(grid) - U_exact(grid)))
print(f"  max |series - exact| on [-1,1]               = {err_series:.3e}")

# ML-style: introduce a convergence-control parameter h on a 1-term homotopy approx
# U_approx(z;h) = c + (-k)*tanh(h*z); optimise h to minimise the PDE residual (no exact soln needed)
def residual_L2(h):
    Z = grid
    Ua = c0 + (-k0)*np.tanh(h*Z)
    dU = -k0*h/np.cosh(h*Z)**2
    d2U = 2*k0*h**2*np.tanh(h*Z)/np.cosh(h*Z)**2
    R = nu0*d2U - (Ua-c0)*dU              # PDE residual of the reduced ODE
    return np.trapezoid(R**2, Z)
opt = minimize_scalar(residual_L2, bounds=(0.2, 3.0), method='bounded')
h_star = opt.x
true_h = k0/(2*nu0)
print(f"  ML-style residual-minimising convergence parameter h* = {h_star:.6f}")
print(f"  exact optimal h (= k/2nu)                              = {true_h:.6f}")
print(f"  |h* - exact|                                          = {abs(h_star-true_h):.3e}")
print("\nDONE.")
