# ============================================================================
# TEST SUITE for symm/ml_pde_solver_demo.py
# Exercises the CONTRACTS of the demo, not just the happy path:
#   T1 determinism given a fixed seed (reproducible baseline-vs-symmetry compare)
#   T2 the exact AZ oracle really solves Fisher-KPP (independent sympy check)
#   T3 null-by-default: result dict + label + honesty guards pass for analysis=None
#   T4 honesty guard is NON-VACUOUS: it FIRES if a "null" interface secretly cheats
#   T5 a nonzero prior flows through autograd with its exact derivative
#   T6 pluggability: a concrete AnalysisInterface subclass (1-D invariant z-chart)
#      runs through the SAME solver body with zero edits and converges
# Run:  python symm/test_ml_pde_solver.py     (from D:/MATHs/math 4890 LieML)
# ============================================================================
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import ml_pde_solver_demo as M
from ml_pde_solver_demo import AnalysisInterface, solve, make_net, u_exact, C

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

print("=" * 70)
print(" TEST SUITE  --  ml_pde_solver_demo.py")
print("=" * 70)

# Fast budget for contract tests (accuracy claim itself is the full-budget run).
FAST = dict(adam_iters=400, lbfgs_iters=120, verbose=False)

# ---------------------------------------------------------------------------
print("\n T1  determinism given a fixed seed")
torch.manual_seed(0); r1 = solve(analysis=None, **FAST)
torch.manual_seed(0); r2 = solve(analysis=None, **FAST)
check("T1 same seed -> identical rel_L2", r1["rel_L2"] == r2["rel_L2"],
      f"{r1['rel_L2']:.6e} == {r2['rel_L2']:.6e}")
check("T1 same seed -> identical res_rms", r1["res_rms"] == r2["res_rms"],
      f"{r1['res_rms']:.6e}")
check("T1 nparams stable", r1["nparams"] == r2["nparams"] == 1341, f"{r1['nparams']}")

# ---------------------------------------------------------------------------
print("\n T2  the exact AZ oracle solves Fisher-KPP (independent sympy proof)")
import sympy as sp
xx, tt = sp.symbols('x t', real=True)
cc = sp.Rational(5, 1) / sp.sqrt(6)
zz = xx - cc * tt
uu = 1 / (1 + sp.exp(zz / sp.sqrt(6))) ** 2
full = sp.simplify(sp.diff(uu, tt) - sp.diff(uu, xx, 2) - uu * (1 - uu))
check("T2 full PDE residual of u_exact == 0", full == 0, f"sympy -> {full}")
# reduced ODE U'' + cU' + U(1-U) with U(z)=1/(1+e^{z/sqrt6})^2
zsym = sp.symbols('z', real=True)
U = 1 / (1 + sp.exp(zsym / sp.sqrt(6))) ** 2
red = sp.simplify(sp.diff(U, zsym, 2) + cc * sp.diff(U, zsym) + U * (1 - U))
check("T2 reduced ODE residual == 0", red == 0, f"sympy -> {red}")
# the code's float speed matches 5/sqrt6
check("T2 code C == 5/sqrt6", abs(C - 5.0 / math.sqrt(6.0)) < 1e-15, f"C={C:.6f}")

# ---------------------------------------------------------------------------
print("\n T3  null-by-default contract")
torch.manual_seed(0); r = solve(analysis=None, **FAST)
check("T3 null_interface flag True", r["null_interface"] is True)
check("T3 input dim == 2 (raw x,t)", r["in_dim"] == 2)
check("T3 data_source disclosed", r["data_source"] == "exact_AZ_oracle")
# default-constructed interface really is the four absences
a = AnalysisInterface()
xg = torch.rand(8, 1); tg = torch.rand(8, 1)
check("T3 coords is identity cat([x,t])",
      torch.equal(a.coords(xg, tg), torch.cat([xg, tg], dim=1)))
check("T3 prior is exactly zero",
      torch.count_nonzero(a.prior(xg, tg)).item() == 0)
check("T3 symmetry_loss is exactly 0",
      float(a.symmetry_loss(make_net(in_dim=2), xg, tg)) == 0.0)
check("T3 certificate is None (null)", a.certificate() is None)
check("T3 feature_dim == 2", a.feature_dim == 2)

# ---------------------------------------------------------------------------
print("\n T4  honesty guard is NON-VACUOUS (fires when a 'null' interface cheats)")
# Monkeypatch the base prior so the default-constructed null interface secretly
# returns a nonzero prior; solve(analysis=None) must REFUSE via the assert.
orig_prior = AnalysisInterface.prior
try:
    AnalysisInterface.prior = lambda self, x, t: torch.ones_like(x)   # cheat
    raised = False
    try:
        torch.manual_seed(0); solve(analysis=None, adam_iters=1, lbfgs_iters=1, verbose=False)
    except AssertionError as e:
        raised = True
        msg = str(e)
    check("T4 guard fires on smuggled nonzero prior", raised,
          msg if raised else "NO AssertionError raised -- guard is vacuous!")
finally:
    AnalysisInterface.prior = orig_prior
# sanity: with the guard restored, null runs clean again
try:
    torch.manual_seed(0); solve(analysis=None, adam_iters=1, lbfgs_iters=1, verbose=False)
    restored_ok = True
except AssertionError:
    restored_ok = False
check("T4 guard restored -> null runs clean", restored_ok)

# ---------------------------------------------------------------------------
print("\n T5  a nonzero prior flows through autograd with its EXACT derivative")
class LinearPrior(AnalysisInterface):
    # u = (2x + 3t) + net(coords);  d(prior)/dx = 2, d(prior)/dt = 3 exactly
    def prior(self, x, t):
        return 2.0 * x + 3.0 * t
ap = LinearPrior()
torch.manual_seed(1)
net = make_net(in_dim=2)
x = torch.rand(16, 1, requires_grad=True)
t = torch.rand(16, 1, requires_grad=True)
# with prior
u = ap.prior(x, t) + net(ap.coords(x, t))
ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
# net-only (fresh graph)
x2 = x.detach().clone().requires_grad_(True)
t2 = t.detach().clone().requires_grad_(True)
u0 = net(torch.cat([x2, t2], 1))
u0x = torch.autograd.grad(u0, x2, torch.ones_like(u0), create_graph=True)[0]
u0t = torch.autograd.grad(u0, t2, torch.ones_like(u0), create_graph=True)[0]
dx = (ux - u0x)            # should be exactly 2
dt = (ut - u0t)            # should be exactly 3
check("T5 d(prior)/dx contributes exactly 2",
      torch.allclose(dx, torch.full_like(dx, 2.0), atol=1e-10),
      f"mean={dx.mean().item():.6f}")
check("T5 d(prior)/dt contributes exactly 3",
      torch.allclose(dt, torch.full_like(dt, 3.0), atol=1e-10),
      f"mean={dt.mean().item():.6f}")

# ---------------------------------------------------------------------------
print("\n T6  pluggability: a 1-D invariant z-chart subclass runs the SAME solver")
class InvariantChart(AnalysisInterface):
    """CCS-SYM+-style symmetry arm: travelling-wave invariant chart z = x - C t.
    Overriding feature_dim + coords routes the SAME solve() through a 1-D network
    with ZERO edits to the solver body (the seam under test)."""
    feature_dim = 1
    def coords(self, x, t):
        return x - C * t                       # (N,1), graph-connected to x,t
    def certificate(self):
        return {"flag": "DEMO", "chart": "z = x - C t", "feature_dim": 1}
torch.manual_seed(0)
rs = solve(analysis=InvariantChart(), **FAST)
check("T6 symmetry arm ran (in_dim == 1)", rs["in_dim"] == 1, f"in_dim={rs['in_dim']}")
check("T6 symmetry arm not flagged null", rs["null_interface"] is False)
check("T6 symmetry arm converged (rel_L2 < 5e-2)", rs["rel_L2"] < 5e-2,
      f"rel_L2={rs['rel_L2']:.3e}")
# same solver body, different injected object -> head-to-head is meaningful
check("T6 baseline & symmetry share matched budget (Nf,Nic,Nbc)",
      (rs["Nf"], rs["Nic"], rs["Nbc"]) == (r1["Nf"], r1["Nic"], r1["Nbc"]),
      f"{(rs['Nf'], rs['Nic'], rs['Nbc'])}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f" RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print(" FAILED: " + ", ".join(FAIL))
print("=" * 70)
sys.exit(1 if FAIL else 0)
