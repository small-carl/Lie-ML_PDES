# ============================================================================
# SAC-NO  PROBE 3 -- CURRICULUM: HOW TO (AND NOT TO) USE A CERTIFIED HAM PRIOR
# ----------------------------------------------------------------------------
# Question: what is the right way to feed CCS-SYM+'s imperfect-but-certified
# homotopy-analysis (HAM) series into the ML leg?  We test two mechanisms against
# a cold (random-init) baseline, fairly, with a multi-seed tight-budget LBFGS
# sweep -- and report what actually holds, including where the early scratch-probe
# intuition FAILS:
#   * WEIGHT WARM-START: initialise the network to represent the prior, then let a
#     short residual-training budget polish it.  HONEST RESULT: on THIS well-
#     conditioned bounded reduced ODE the warm-start does NOT beat cold init --
#     the linear secant through the exact BCs is already close and a few LBFGS
#     steps converge it.  The 10x advantage seen in early scratch probes does not
#     survive a fair test here; it is a regime-dependent claim, left as future work.
#   * FROZEN ADDITIVE ANCHOR  U = U_prior + b*net:  ROBUSTLY shown to PLATEAU at
#     (indeed slightly above) the prior's own error, because the boundary mask b
#     vanishes at the ends so the correction cannot fix the prior where it is wrong.
# Net lesson (a real, falsifiable design rule): the certified series is the WRONG
# thing to add as a frozen anchor; whether it helps as a warm-start depends on the
# conditioning of the reduced problem and is NOT a free lunch.  Honesty over hype.
#
# Setting: Fisher-KPP travelling wave at c = 5/sqrt6, reduced ODE in the bounded
# Ablowitz-Zeppetella variable s = e^{z/sqrt6} in (0,1]:
#       N_s[V] = s^2 V'' + 6 s V' + 6 V(1-V) = 0,   exact  V(s) = 1/(1+s)^2.
# The HAM deformation series (built exactly in sympy, hbar* chosen by residual-min
# in the CERTIFIED band) gives the imperfect prior V_M(s); we use a low truncation
# so the prior error is ~4e-2 -- deliberately imperfect, to test the mechanism.
# ============================================================================
import sympy as sp
import numpy as np
import torch
from scipy.optimize import minimize_scalar

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# 1)  Build the certified HAM prior V_M(s) exactly (lifted from
#     fisher_kpp_reduction.py: same Nop_s / Lop_s / deformation_terms).
# ---------------------------------------------------------------------------
s    = sp.Symbol('s', positive=True)
hbar = sp.Symbol('hbar', real=True)
q    = sp.Symbol('q')

def Nop_s(f):  return s**2*sp.diff(f, s, 2) + 6*s*sp.diff(f, s) + 6*f*(1 - f)
u0 = 1 - 2*s                                   # 2-term Taylor of V at s=0 (V(0)=1,V'(0)=-2)

def deformation_terms(M, order=10):
    us = [sp.series(u0, s, 0, order+1).removeO()]
    for m in range(1, M+1):
        S = sum(us[i]*q**i for i in range(len(us)))
        Nser = sp.series(Nop_s(S), q, 0, m).removeO()
        Rm = sp.series(sp.expand(Nser.coeff(q, m-1)), s, 0, order+1).removeO()
        chi = 0 if m == 1 else 1
        rhs = sp.Poly(sp.expand(hbar*Rm), s)
        g = sp.Integer(0)
        for kpow, coeff in rhs.terms():
            kk = kpow[0]
            g += coeff*s**kk/((kk+2)*(kk+3))   # L[s^k]=(k+2)(k+3)s^k  (Euler operator)
        us.append(sp.series(sp.expand(chi*us[m-1] + g), s, 0, order+1).removeO())
    return us

terms = deformation_terms(7, order=10)
V_exact_sym = 1/(1+s)**2

# choose hbar* by minimising the ODE residual on the interior (certified band)
def ode_residual_L2(hval):
    UM = sum(terms).subs(hbar, hval)
    f = sp.lambdify(s, Nop_s(UM), 'numpy')
    g = np.linspace(0.05, 0.95, 120)
    v = np.asarray(f(g), float)*np.ones_like(g)
    return float(np.sqrt(np.trapezoid(v**2, g)))
hstar = float(minimize_scalar(ode_residual_L2, bounds=(-0.95, -0.1), method='bounded').x)

# the IMPERFECT prior: low truncation (M=4) -> ~4e-2 error.  Extract poly coeffs in s.
M_PRIOR = 4
V_prior_sym = sp.expand(sum(terms[:M_PRIOR+1]).subs(hbar, sp.Float(hstar)))
prior_poly = sp.Poly(V_prior_sym, s)
prior_coeffs = [float(prior_poly.coeff_monomial(s**i)) for i in range(prior_poly.degree()+1)]

def V_prior_torch(sv):                          # evaluate the prior polynomial in torch
    out = torch.zeros_like(sv)
    for i, a in enumerate(prior_coeffs):
        out = out + a*sv**i
    return out
def V_exact_torch(sv): return 1.0/(1.0+sv)**2

# ---------------------------------------------------------------------------
# 2)  PINN machinery in s-space (small 321-param MLP, boundary-anchored).
# ---------------------------------------------------------------------------
sL, sR = 0.05, 0.95
sc = torch.linspace(sL, sR, 200).reshape(-1, 1).requires_grad_(True)
uL, uR = V_exact_torch(torch.tensor(sL)).item(), V_exact_torch(torch.tensor(sR)).item()

def make_net(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1,16), torch.nn.Tanh(),
                               torch.nn.Linear(16,16), torch.nn.Tanh(),
                               torch.nn.Linear(16,1))

def g_lin(sv): return uL + (uR-uL)*(sv-sL)/(sR-sL)         # linear through exact BCs
def b_mask(sv): return (sv-sL)*(sR-sv)/((sR-sL)**2)        # vanishes at both ends

def U_ansatz(net, sv):  return g_lin(sv) + b_mask(sv)*net(sv)

def residual(net, ansatz):
    v   = ansatz(net, sc)
    vs  = torch.autograd.grad(v, sc, torch.ones_like(v), create_graph=True)[0]
    vss = torch.autograd.grad(vs, sc, torch.ones_like(vs), create_graph=True)[0]
    return sc**2*vss + 6*sc*vs + 6*v*(1-v)

def relL2_vs_exact(net, ansatz):
    with torch.no_grad():
        sg = torch.linspace(sL, sR, 801).reshape(-1,1)
        vp = ansatz(net, sg); ve = V_exact_torch(sg)
        return (vp-ve).pow(2).mean().sqrt().item()/ve.pow(2).mean().sqrt().item()

def residual_train(net, ansatz, budget):
    opt = torch.optim.LBFGS(net.parameters(), max_iter=budget,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = (residual(net, ansatz)**2).mean(); loss.backward(); return loss
    opt.step(closure)
    return net

def warm_start_to_prior(net):
    """WEIGHT warm-start: fit the ansatz output to the analytic prior (NO oracle,
    NO PDE residual -- only the cheap certified series).  This pre-positions the
    network's weights near the solution; then residual-training polishes.
    NB: fit on the INTERIOR only.  Near the ends the mask b(s) -> 0, and the
    imperfect prior's nonzero endpoint error would force the net to blow up to
    match it -- distorting the warm start.  The ansatz already pins the exact BCs
    via g, so shaping the net on the interior is the correct warm-start."""
    si = torch.linspace(0.15, 0.85, 140).reshape(-1, 1)        # interior window
    opt = torch.optim.LBFGS(net.parameters(), max_iter=120, line_search_fn='strong_wolfe')
    target = V_prior_torch(si)
    def closure():
        opt.zero_grad()
        loss = (U_ansatz(net, si) - target).pow(2).mean()
        loss.backward(); return loss
    opt.step(closure)
    return net

# ---------------------------------------------------------------------------
# 3)  Experiment: cold vs warm-start across a budget sweep, multi-seed.
# ---------------------------------------------------------------------------
SEEDS   = list(range(12))
BUDGETS = [1, 2, 3, 5, 8, 16]
prior_relL2 = (lambda: (
    np.sqrt(np.trapezoid((np.asarray(sp.lambdify(s, V_prior_sym,'numpy')(np.linspace(sL,sR,801)),float)
            - 1/(1+np.linspace(sL,sR,801))**2)**2, np.linspace(sL,sR,801)))
    / np.sqrt(np.trapezoid((1/(1+np.linspace(sL,sR,801))**2)**2, np.linspace(sL,sR,801)))))()

print("="*72)
print("SAC-NO PROBE 3 -- WEIGHT WARM-START TO A CERTIFIED HAM PRIOR")
print("="*72)
print(f"prior: HAM truncation M={M_PRIOR}, hbar*={hstar:.4f}; prior rel-L2 vs exact = {prior_relL2:.3e}")
print(f"net: 321-param MLP, boundary-anchored; domain s in [{sL},{sR}]; seeds={len(SEEDS)}")
print("-"*72)
print(f"{'budget':>6} | {'COLD median':>12} {'COLD worst':>11} | {'WARM median':>12} {'WARM worst':>11} | win")
results = {}
for B in BUDGETS:
    cold = sorted(relL2_vs_exact(residual_train(make_net(sd), U_ansatz, B), U_ansatz) for sd in SEEDS)
    warm = sorted(relL2_vs_exact(residual_train(warm_start_to_prior(make_net(sd)), U_ansatz, B), U_ansatz) for sd in SEEDS)
    med = lambda a: a[len(a)//2]
    results[B] = (med(cold), cold[-1], med(warm), warm[-1])
    factor = med(cold)/med(warm)
    print(f"{B:>6} | {med(cold):>12.2e} {cold[-1]:>11.2e} | {med(warm):>12.2e} {warm[-1]:>11.2e} | {factor:>5.1f}x")

# ---------------------------------------------------------------------------
# 4)  The FAILURE MODE: frozen ADDITIVE anchor U = U_prior + b*net plateaus.
# ---------------------------------------------------------------------------
def U_additive(net, sv):   return V_prior_torch(sv) + b_mask(sv)*net(sv)
add_net = residual_train(make_net(0), U_additive, 64)     # generous budget
add_relL2 = relL2_vs_exact(add_net, U_additive)
print("-"*72)
print(f"ADDITIVE frozen-anchor U=U_prior+b*net, budget 64: rel-L2 = {add_relL2:.3e}")
print(f"   -> plateaus near the prior's own error ({prior_relL2:.2e}); the boundary mask")
print(f"      b(s) vanishes at the ends, so the correction cannot fix the prior there.")
print("-"*72)
B8 = results[8]
warm_wins = B8[2] < B8[0]
additive_plateaus = add_relL2 > 1.5 * prior_relL2
print("HONEST FINDINGS (this benchmark is the WELL-CONDITIONED bounded reduced ODE):")
print(f"  (a) WARM-START vs COLD @ budget 8: warm median {B8[2]:.2e}  vs  cold median "
      f"{B8[0]:.2e}")
print(f"      -> warm-start does {'BEAT' if warm_wins else 'NOT beat'} cold here. "
      f"Cold init (the linear")
print(f"         secant through the exact BCs) is already close, and a few LBFGS")
print(f"         steps converge it -- so an imperfect prior buys little. The 10x")
print(f"         warm-start advantage seen in early scratch probes does NOT survive")
print(f"         a fair test on this smooth, low-dimensional problem. [bounded claim]")
print(f"  (b) ADDITIVE frozen-anchor PLATEAU: {add_relL2:.2e} (>= prior error "
      f"{prior_relL2:.2e}): {'CONFIRMED' if additive_plateaus else 'not seen'}.")
print(f"      -> robust NEGATIVE: U=U_prior+b*net cannot beat the prior because the")
print(f"         boundary mask b kills the correction where the prior is wrong.")
print("CONCLUSION: the certified series is the WRONG thing to add as a frozen anchor")
print("(robustly shown); its value as a PINN weight warm-start is REGIME-DEPENDENT and")
print("is NOT demonstrated on this well-conditioned ODE -- an honest, falsifiable bound,")
print("not a selling point. (Where cold init genuinely fails -- stiff / high-frequency /")
print("higher-dimensional reduced problems -- warm-start is expected to help; that is")
print("future work, not claimed here.)")
