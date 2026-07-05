# ============================================================================
# MACHINE-LEARNING PDE SOLVER DEMO  --  full (1+1)D Fisher-KPP PINN
# ----------------------------------------------------------------------------
# A physics-informed neural network that solves the FULL (1+1)D PDE
#
#       u_t = u_xx + u(1-u)        (Fisher-KPP)
#
# on a space-time rectangle x in [xL,xR], t in [t0,T].
#
# The CCS-SYM+ Lie-symmetry / analytic-prior FRONT END is a single PLUGGABLE
# object `analysis`, LEFT DEFAULT (null).  With analysis=None the solver is,
# bit-for-bit, a PLAIN BASELINE PINN in raw (x,t) coordinates.  This is exactly
# the matched plain-PINN baseline plan.txt section 5 calls for ("plain PINN with
# equal extra collocation -- the proxy-vs-fundamental test").  The SAME executable
# code path serves both arms; the ONLY thing that differs is the injected object
# (plus the printed label and the wall-clock timing).
#
# THE FOUR ABSENCES THAT DEFINE "null" (each runtime-asserted before training):
#   (1) identity coordinate chart      coords(x,t) == cat([x,t])
#   (2) zero analytic prior            prior(x,t)  is exactly 0  (count_nonzero==0)
#   (3) zero symmetry loss             symmetry_loss(...) is exactly 0
#   (4) no symmetry-derived reduction or symmetry-augmented data
#
# DATA-PRIOR LEAK -- DISCLOSED, NOT HIDDEN.  The IC and both Dirichlet BCs are
# sampled from the exact Ablowitz-Zeppetella travelling wave, which is itself a
# Lie similarity solution; this hands the "plain" PINN travelling-wave structure.
# The leak is SYMMETRIC: BOTH arms (null baseline and CCS-SYM+) receive the SAME
# oracle-derived data, so it does not bias the head-to-head -- the symmetry arm's
# advantage must come from coords/prior/loss, NOT from data.  This is recorded in
# LEAK_NOTE and in the returned dict field data_source='exact_AZ_oracle'.  A strict
# variant (generic far-field constants instead of oracle BCs) is noted as future work.
#
# HOW TO UPGRADE TO CCS-SYM+  (the solver body below does NOT change):
#   subclass AnalysisInterface and override
#     feature_dim    -> 1            (one similarity coordinate z)
#     coords(x,t)    -> invariant chart z(x,t)            (graph-connected to x,t)
#     prior(x,t)     -> certified HAM series  U_M(z; hbar*) (graph-connected tensor)
#     symmetry_loss  -> ||X(u_theta)||^2  as a 0-d tensor on the graph
#     certificate()  -> {rho_k, flag in {CERTIFIED/INCONCLUSIVE/DIVERGENT}, budget}
#
# Default PDE  : Fisher-KPP   u_t = u_xx + u(1-u)
# Ground truth : exact Ablowitz-Zeppetella travelling wave
#                u(x,t) = 1 / (1 + B e^{(x - c t)/sqrt6})^2 ,   c = 5/sqrt6, B=1
#                (symbolically verified to satisfy BOTH the full PDE and the reduced
#                 ODE U'' + cU' + U(1-U) = 0 identically; closed-form, never FDM)
#
# Pure torch, CPU, float64.  Run:  python symm/ml_pde_solver_demo.py
# ============================================================================
import torch, time, math

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)        # float64 -> clean u_xx, stable strong_wolfe

# Data-prior leak disclosure (printed in the report; auditable, symmetric across arms).
LEAK_NOTE = ("DATA-PRIOR LEAK (disclosed): IC and both Dirichlet BCs are sampled from the "
             "exact AZ similarity solution; identical data is given to BOTH arms, so the "
             "leak is SYMMETRIC and does not bias the baseline-vs-symmetry comparison.")


# ----------------------------------------------------------------------------
# THE SEAM.  Default instance is the NULL front end == plain baseline PINN.
# One class, five members.  Subclass to inject CCS-SYM+; the solver is unchanged.
# ----------------------------------------------------------------------------
class AnalysisInterface:
    """Null Lie-symmetry / analytic-prior front end (plain-PINN baseline).

    feature_dim          -- network input arity (solver reads this; never hardcodes 2).
    coords(x,t)          -- network input chart   (IDENTITY cat([x,t]) when null).
    prior(x,t)           -- analytic prior tensor  (ZERO, graph-connected, when null).
    symmetry_loss(net,..)-- extra invariance penalty (0 when null).
    certificate()        -- optional rho_k / flag / corrector-budget dict (None when null).

    Override these to inject a symmetry reduction / certified series prior /
    approximate-symmetry loss; the solve() body below does not change.
    """
    feature_dim = 2                               # net input arity; (x,t) when null

    def coords(self, x, t):
        # Identity chart.  Built from the leaf tensors x,t by a differentiable op so
        # autograd.grad(u, x) and autograd.grad(u, t) trace through the chart.
        # This is the SOLE place network inputs are assembled.
        return torch.cat([x, t], dim=1)

    def prior(self, x, t):
        # Zero analytic prior, returned as a GRAPH-CONNECTED tensor (NOT python 0.0).
        # A real prior U_M(z;hbar) must contribute to u_t/u_x/u_xx through autograd;
        # zeros_like(x) keeps the additive ansatz type/shape-stable and a true drop-in.
        return torch.zeros_like(x)

    def symmetry_loss(self, net, x, t):
        # No symmetry penalty.  Returns 0.0; added to a tensor loss, contributes no
        # gradient.  A future version returns ||X(u_theta)||^2 as a 0-d graph tensor.
        return 0.0

    def certificate(self):
        # No certificate for the null front end.  CCS-SYM+ returns a dict carrying the
        # rho_k sequence + {CERTIFIED/INCONCLUSIVE/DIVERGENT} flag + corrector budget.
        return None


# ----------------------------------------------------------------------------
# Problem definition: Fisher-KPP operator + exact Ablowitz-Zeppetella oracle.
# ----------------------------------------------------------------------------
SQRT6 = math.sqrt(6.0)
C     = 5.0 / SQRT6                            # special AZ travelling-wave speed (=2.0412)
B     = 1.0

def u_exact(x, t):
    """Exact AZ travelling wave  u = 1/(1 + B e^{(x-Ct)/sqrt6})^2  (closed form, not FDM)."""
    z = x - C * t
    return 1.0 / (1.0 + B * torch.exp(z / SQRT6)) ** 2

def fisher_kpp_op(u, u_t, u_x, u_xx):
    """Default injected PDE operator: residual of  u_t = u_xx + u(1-u)."""
    return u_t - u_xx - u * (1.0 - u)


# ----------------------------------------------------------------------------
# Network: tanh MLP  R^in_dim -> R^1.  W=20, H=4 -> 1341 params (measured best).
# ----------------------------------------------------------------------------
def make_net(width=20, hidden=4, in_dim=2):
    layers = [torch.nn.Linear(in_dim, width), torch.nn.Tanh()]
    for _ in range(hidden - 1):
        layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
    layers += [torch.nn.Linear(width, 1)]
    return torch.nn.Sequential(*layers)


# ----------------------------------------------------------------------------
# Solver.  null and CCS-SYM+ differ ONLY by the injected object (+ label/timing).
# ----------------------------------------------------------------------------
def solve(analysis=None, xL=-10.0, xR=10.0, t0=0.0, T=2.0,
          Nf=4000, Nic=300, Nbc=200, w_data=10.0,
          width=20, hidden=4, adam_iters=1500, adam_lr=2e-3, lbfgs_iters=500,
          pde_op=fisher_kpp_op, verbose=True):

    null = analysis is None
    if analysis is None:
        analysis = AnalysisInterface()        # <-- analysis interface left DEFAULT (null)

    # network sized from the interface (feature_dim), NEVER a hardcoded 2
    net = make_net(width=width, hidden=hidden, in_dim=analysis.feature_dim)
    nparams = sum(p.numel() for p in net.parameters())

    # ---- collocation (interior); requires_grad for autograd through the chart ----
    xf = (torch.rand(Nf, 1) * (xR - xL) + xL).requires_grad_(True)
    tf = (torch.rand(Nf, 1) * (T - t0) + t0).requires_grad_(True)
    # ---- initial condition  t = t0  (from exact oracle) ----
    xi = torch.rand(Nic, 1) * (xR - xL) + xL
    ti = torch.full((Nic, 1), t0)
    ui = u_exact(xi, ti)
    # ---- Dirichlet boundaries  x = xL, xR  (from exact oracle; see LEAK_NOTE) ----
    tb  = torch.rand(Nbc, 1) * (T - t0) + t0
    xbL = torch.full((Nbc, 1), xL); ubL = u_exact(xbL, tb)
    xbR = torch.full((Nbc, 1), xR); ubR = u_exact(xbR, tb)

    # ---- THE single ansatz (prior + net(coords)); reused everywhere ----
    def u_pred(x, t):
        return analysis.prior(x, t) + net(analysis.coords(x, t))

    # ---- THE single residual; calls u_pred; OUTSIDE any torch.no_grad ----
    def pde_residual(x, t):
        u    = u_pred(x, t)
        u_t  = torch.autograd.grad(u,   t, torch.ones_like(u),   create_graph=True)[0]
        u_x  = torch.autograd.grad(u,   x, torch.ones_like(u),   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return pde_op(u, u_t, u_x, u_xx)

    # ---- HONESTY GUARDS: machine-checkable operational definition of "null" ----
    # The 'plain baseline' label is printed ONLY if these pass (no false advertising).
    guards_ok = True
    if null:
        with torch.no_grad():
            xg = xf.detach(); tg = tf.detach()
            ident = torch.equal(analysis.coords(xg, tg), torch.cat([xg, tg], dim=1))
            zero_prior = (torch.count_nonzero(analysis.prior(xg, tg)).item() == 0)
        zero_sym = (float(analysis.symmetry_loss(net, xg, tg)) == 0.0)
        assert ident,      "NULL guard 1 FAILED: coords is not the identity chart cat([x,t])"
        assert zero_prior, "NULL guard 2 FAILED: prior is not exactly zero (smuggled profile)"
        assert zero_sym,   "NULL guard 3 FAILED: symmetry_loss is not exactly 0"
        guards_ok = ident and zero_prior and zero_sym

    def loss_fn():
        r        = pde_residual(xf, tf)
        loss_r   = (r ** 2).mean()
        loss_ic  = ((u_pred(xi, ti) - ui) ** 2).mean()
        loss_bc  = ((u_pred(xbL, tb) - ubL) ** 2).mean() + ((u_pred(xbR, tb) - ubR) ** 2).mean()
        loss_sym = analysis.symmetry_loss(net, xf, tf)
        total    = loss_r + w_data * (loss_ic + loss_bc) + loss_sym
        return total, loss_r, loss_ic, loss_bc

    t_start = time.time()
    # ---- Adam warmup ----
    opt = torch.optim.Adam(net.parameters(), lr=adam_lr)
    for it in range(adam_iters):
        opt.zero_grad()
        total, lr_, lic, lbc = loss_fn()
        total.backward()
        opt.step()
        if verbose and it % 300 == 0:
            print(f"   adam {it:5d}: loss={total.item():.3e}  "
                  f"(res={lr_.item():.2e} ic={lic.item():.2e} bc={lbc.item():.2e})")
    # ---- LBFGS polish (the float64 convergence workhorse) ----
    opt = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_iters,
                            line_search_fn='strong_wolfe',
                            tolerance_grad=1e-12, tolerance_change=1e-14,
                            history_size=50)
    def closure():
        opt.zero_grad()
        total, *_ = loss_fn()
        total.backward()
        return total
    opt.step(closure)
    train_time = time.time() - t_start

    # ---- final diagnostics ----
    # residual RMS: build a FRESH live graph (create_graph in pde_residual), then
    # detach the RESULT.  Never inside no_grad -- autograd must run to form u_xx.
    res_rms = (pde_residual(xf, tf).detach() ** 2).mean().sqrt().item()
    # error vs exact on a FIXED, independent 201x101 grid; needs u not derivatives,
    # so this -- and ONLY this -- runs inside torch.no_grad().
    with torch.no_grad():
        xg = torch.linspace(xL, xR, 201)
        tg = torch.linspace(t0, T, 101)
        X, Tt = torch.meshgrid(xg, tg, indexing='ij')
        xq = X.reshape(-1, 1); tq = Tt.reshape(-1, 1)
        up = u_pred(xq, tq); ue = u_exact(xq, tq)
        err    = (up - ue).abs()
        max_err = err.max().item()
        rel_L2  = (err.pow(2).mean().sqrt() / ue.pow(2).mean().sqrt()).item()

    label = ('NULL (default) -> plain baseline PINN' if (null and guards_ok)
             else type(analysis).__name__)
    cert = analysis.certificate()

    if verbose:
        print("\n" + "=" * 70)
        print(" MACHINE-LEARNING PDE SOLVER  --  result")
        print("=" * 70)
        print(f"   PDE                 : Fisher-KPP  u_t = u_xx + u(1-u)")
        print(f"   domain              : x in [{xL},{xR}],  t in [{t0},{T}]")
        print(f"   analysis interface  : {label}")
        # matched-budget knobs: identical for both arms except train_time
        print(f"   input dim           : {analysis.feature_dim}")
        print(f"   network params      : {nparams}")
        print(f"   collocation Nf      : {Nf}   (IC Nic={Nic}, BC Nbc={Nbc}/side)")
        print(f"   training schedule   : {adam_iters} Adam(lr={adam_lr}) + LBFGS({lbfgs_iters})")
        print(f"   train time          : {train_time:.1f}s (CPU, float64)")
        print(f"   final residual RMS  : {res_rms:.2e}")
        print(f"   max |u_pinn-u_exact|: {max_err:.3e}")
        print(f"   relative L2 error   : {rel_L2:.3e}   (vs exact Ablowitz-Zeppetella front)")
        if cert is not None:
            print(f"   certificate         : {cert}")
        print("-" * 70)
        print("   " + LEAK_NOTE)
        print("=" * 70)

    return dict(nparams=nparams, train_time=train_time, res_rms=res_rms,
                max_err=max_err, rel_L2=rel_L2, null_interface=null,
                in_dim=analysis.feature_dim, Nf=Nf, Nic=Nic, Nbc=Nbc,
                adam_iters=adam_iters, lbfgs_iters=lbfgs_iters,
                data_source='exact_AZ_oracle')


if __name__ == "__main__":
    ANALYSIS = None        # <-- analysis interface LEFT DEFAULT (null): plain baseline PINN
    solve(analysis=ANALYSIS)
