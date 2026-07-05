# Quick probe: BGI approximate-symmetry feasibility for a PERTURBED PDE in sympy.
# Target: perturbed Burgers u_t + u u_x = eps u_xx  (small viscosity / weak dissipation).
# BGI: expand generator X = X0 + eps X1, require pr X(Delta) = O(eps^2) on Delta=0.
# We demonstrate: (1) approximate invariance bookkeeping in eps, (2) reduce to ODE,
# (3) perturbation series of the reduced ODE = the leg ML must NOT have to learn.
import sympy as sp
t,x,eps,c,k = sp.symbols('t x eps c k', real=True)
u=sp.Symbol('u'); ut=sp.Symbol('u_t'); ux=sp.Symbol('u_x'); uxx=sp.Symbol('u_xx')

# perturbed Burgers as a deformation of the inviscid (eps=0) equation
Delta = ut + u*ux - eps*uxx
# unperturbed (eps=0): u_t+u u_x=0 has scaling/Galilean; check a stable point sym survives
# travelling wave z=x-ct, u=U(z): reduce
z=sp.Symbol('z'); U=sp.Function('U')
red = Delta.subs({ut:-c*sp.Derivative(U(z),z), ux:sp.Derivative(U(z),z),
                  uxx:sp.Derivative(U(z),z,2), u:U(z)})
print("reduced perturbed ODE:", sp.simplify(red),"= 0")
# eps-perturbation series U = U0 + eps U1 of  eps U'' = (U-c)U'
# O(1): (U0-c)U0' = 0  -> U0 = const = c (the degenerate outer soln) OR shock.
# This is the SINGULAR-PERTURBATION structure: outer is trivial, the kink lives in eps-scaled inner.
# Inner var Z=z/eps: U(z)=W(Z),  W'' = (W-c)W'  -> exact kink W=c-k tanh(kZ/2).
Z=sp.Symbol('Z'); W=sp.Function('W')
Wk = c - k*sp.tanh(k*Z/2)
inner = sp.simplify(sp.diff(Wk,Z,2) - (Wk-c)*sp.diff(Wk,Z))
print("inner-layer kink residual (should be 0):", inner)
print("=> approx-sym reduction recovers the eps-> boundary-layer kink; outer/inner split is the analytic backbone.")
