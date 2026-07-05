# BUILD PLAN — CCS-SYM+ Measurement & Base-Benchmark Harness

> **What this file is.** A *self-contained, executable build specification* for the
> measurement + base-benchmark system of the CCS-SYM+ research program. It is written
> to be handed to a **fresh agent/session with no memory of the chat that produced it**.
> Read this top to bottom, then build the package described in §4–§9, then run the
> verification gates in §10. Everything you need (the math, the function signatures, the
> expected numbers) is inlined below — but you must still **read the four reference
> scripts** named in §3 to lift the verified engine code verbatim rather than re-deriving it.
>
> **Author's note to the executor:** this is a *measurement yardstick*, not new science.
> Its job is to score, on a fixed and honest scale, the four capabilities the thesis claims
> (reduce → close → certify → correct). The science is already verified (Part II of
> `plan.txt`). Do not "improve" the methodology here; **measure it faithfully**, including
> where it fails (the VACUOUS-bound case is a feature, not a bug — see §7 and §12).

---

## 0. How to launch (read first)

```
# working dir:
cd "D:\MATHs\math 4890 LieML"

# environment (already installed, do NOT reinstall):
#   Python 3.14.2, sympy 1.14, numpy 2.3.5, scipy 1.17.1, torch 2.12.0+cpu
#   Windows 11, PowerShell + Bash both available. NOT a git repo.

# build target dir (this file already lives here):
#   D:\MATHs\math 4890 LieML\symm\bench\

# temp/scratch (use this, not C:\temp or /tmp):
#   C:\Users\Administrator\.claude\jobs\e4d6590d\tmp
```

**Time budget.** This is designed for an **unattended ~8-hour window**. That budget is the
whole reason the lowest solver tier (§2, Level 0) is allowed to run pure-symbolic at
**maximum fidelity** (exact rationals, high HAM/series order, full `sp.solve` of the
determining system, full simplification, generous timeouts). Do not take the seconds-scale
shortcuts the existing demos took — those demos proved feasibility; this harness establishes
*ground truth*. Prioritize per §13 so that even if symbolic stages run long, the scorecard
still completes.

**Done = §10 gates pass and `run_benchmarks.py` writes a scorecard.** When finished, write
`symm/bench/RESULTS.md` summarizing the scorecard and any gate that failed (with the honest
reason, per §12).

---

## 1. Mission

Build the **measure-and-base-benchmark system** that evaluates the *ability* of the CCS-SYM+
methodology before the main Phases 1–6 run. Concretely the harness must, for each benchmark
problem, automatically:

1. **REDUCE** — verify the Lie generator(s) (`invariance_residual == 0`) and reduce the PDE
   to its ODE `N[U]=0` (`back_substitution_residual == 0`).
2. **CLOSE** — build the HAM deformation series `U_M(z;ħ)` on the reduced ODE.
3. **CERTIFY** — emit the three-part certificate (ratio flag + geometric tail; closedness R1;
   computable a-posteriori bound R2) and the metrics of `plan.txt §5`.
4. **CORRECT** — train the small symmetry-invariant PINN and report its accuracy + equivariance.
5. **SCORE** — roll all of the above into per-problem metrics and a single weighted
   **capability scorecard ("ability index")** (§9), with full honest-negative handling.

The output is a reproducible scorecard + per-problem records, **anchored to a Level-0
pure-symbolic ground truth** (§2). This is the yardstick every later phase is graded against.

---

## 2. Solver hierarchy (THE key design directive)

Every quantity the harness reports is produced at one of **three solver levels**, lowest first.
The **lowest level is the authority**; higher levels are scored *against* it.

### Level 0 — Pure Python + sympy symbolic solver, **MAXIMUM RUNTIME** (the ground truth)

This is the foundational tier and the reason for the 8-hour window. It uses **only sympy**
(exact arithmetic, no floats) and is configured for *fidelity over speed*:

- **Exact rationals everywhere.** `c = sp.Rational(5,1)/sp.sqrt(6)`, `ħ` values as
  `sp.Rational`, ODE coefficients exact. No `sp.Float` until the very last lambdify-for-plot step.
- **Full determining-system solve.** For the reduction/auto-discovery checks, build the generic
  polynomial infinitesimal ansatz, prolong, substitute on-shell, **split into the determining
  system, and `sp.solve` it** to recover the algebra dimension — do not hand it the answer.
- **High HAM order.** `M_SYM = 12` deformation terms (the demos used 7). High series order in
  the bounded variable: `ORDER_SYM = 20` (demos used 8–10).
- **Symbolic norms and ratios.** Compute term L2 norms and Turkyilmazoglu ratios `ρ_k` from the
  symbolic series; only convert to float for the final reported number.
- **Full simplification.** Use `sp.simplify`/`sp.factor`/`sp.cancel` as needed; do **not**
  short-circuit with `nsimplify`-only shortcuts.
- **Generous / no timeouts + progress logging.** Wrap each expensive symbolic call so it logs
  start/elapsed (so an 8-hour run is observable). Allow long calls; only fall back to a lower
  fidelity (e.g. `M_SYM=8`, `ORDER_SYM=12`) if a single call exceeds a hard cap (default
  `SYM_CALL_CAP_S = 1800` = 30 min) — and **log the downgrade explicitly** so the scorecard is honest.
- **Determinism.** No randomness at Level 0.

Level 0 is the symbolic oracle: exact closed-form solutions (Cole–Hopf tanh, Barenblatt,
Ablowitz–Zeppetella, Blasius constant via high-order series) and the exact reduced ODEs.

### Level 1 — Numeric scipy/numpy (the working tier)

`solve_ivp` shooting cross-checks, `minimize_scalar` for `ħ*`/`h*`, numeric L2 norms on grids,
numeric ratio sequences. Fast, used for the bulk of the metrics, **always reconciled against
Level 0** (e.g. numeric `ħ*` must land inside the Level-0 CERTIFIED band; numeric reduced-ODE
residual of the Level-0 exact solution must be ~0 to numeric tolerance).

### Level 2 — torch PINN corrector (the ML tier)

The small CPU MLP of Stage S4. Scored against the Level-0 / oracle solution. CPU only,
`torch.manual_seed(0)` for reproducibility.

**Reconciliation rule (must be enforced and reported):** for every benchmark, the harness emits
a `level_agreement` block: `|Level1_value − Level0_value|` for the shared quantities (exact-soln
residual, `ħ*` location, series accuracy at matched order). Large disagreement is a *finding*,
logged, not silently averaged.

---

## 3. Context recovery — read these before building

You have no memory of the originating chat. Recover context from these files (all under
`D:\MATHs\math 4890 LieML\`). **Read them; reuse their verified code verbatim.**

| File | What to lift | Key verified facts/numbers |
|---|---|---|
| `plan.txt` | §2 methodology (S0–S4), §4 benchmark table, §5 metrics, §6 risk table | the spec basis; the certificate definitions R1/R2/R3 are in §2 (S3) |
| `symm/feasibility_demo.py` | the jet engine: `Dx`, `Dt`, `prolong2`, `invariance_residual`; Burgers Δ + 5 generators; tanh kink; porous-medium scaling + Barenblatt | Burgers 5 generators residual **0**; tanh kink residual **0**; Barenblatt residual **0** for m=1,2,3, β=1/(m+2); series max err **1.56e-2** on [−1,1]; `h*` recovers k/2ν to **1.7e-7** |
| `symm/fisher_kpp_reduction.py` | **PRIMARY**: the SAME jet engine (duplicated); `Nop_s`, `Lop_s`, `u0=1-2s`, `deformation_terms(M,order)`, `term_L2_norm`; the bounded variable `s=e^{z/√6}`; the AZ oracle | Fisher symmetries residual **0** + scaling negative control nonzero; reduced ODE `U''+cU'+U(1−U)=0`; AZ `U=1/(1+B e^{z/√6})²` residual **0**; `N_s[V]` for `V=1/(1+s)²` is **0**; eigenvalue `(k+2)(k+3)`; ρ_k **ħ-dependent** (ħ=−1 crosses 1 at **1.105/1.175** → DIVERGENT; ħ=−0.6/−0.3 → CERTIFIED); rel-L2 **1.02e-1 → 4.48e-2 → 2.04e-2** (M=3,5,7); `solve_ivp` backward cross-check **3.6e-10** |
| `symm/pinn_corrector_demo.py` | Stage S4: 321-param MLP, boundary-anchored `U(z)=g(z)+b(z)·net(z)`, LBFGS strong_wolfe max_iter=300, 400 collocation | **321 params**, train **0.13 s**, rel-L2 **8.7e-4**, residual RMS **3.0e-4**. NOTE: `residual()` must be computed OUTSIDE `torch.no_grad()` (it needs autograd) |
| `symm/bgi_probe.py` | perturbed Burgers `u_t+u u_x=ε u_xx`; inner var `Z=z/ε`; inner kink `W=c−k tanh(kZ/2)` residual 0 | inner-layer kink residual **0** (Phase-4 seed; optional for this harness) |

> The jet engine is **duplicated** across `feasibility_demo.py` and `fisher_kpp_reduction.py`.
> Your first build task (`engine.py`) is to **consolidate it into one module** that both the
> old demos and the new harness can import. Copy the version from `fisher_kpp_reduction.py`
> lines ~36–59 (it is identical to `feasibility_demo.py` lines ~24–47).

---

## 4. Package layout to build

Create under `D:\MATHs\math 4890 LieML\symm\bench\`:

```
symm/bench/
  __init__.py
  engine.py            # consolidated jet/prolongation engine (Dx,Dt,prolong2,invariance_residual)
  symbolic.py          # LEVEL 0: max-runtime sympy ground-truth (oracles, exact reductions, det-system solve)
  problems.py          # the benchmark registry: 5 base problems as a uniform spec
  ham.py               # HAM deformation-series builder (generic; bounded-variable aware)
  certify.py           # the 3-part certificate (ratio+tail R3, closedness R1, a-posteriori R2)
  metrics.py           # plan.txt §5 metrics + level-agreement reconciliation
  pinn.py              # LEVEL 2: small symmetry-invariant torch corrector (reusable)
  scorecard.py         # weighted capability "ability index" aggregation
  run_benchmarks.py    # the runner: loops problems, runs all levels, writes records + scorecard
  README.md            # how to run + what each metric means
  RESULTS.md           # WRITE THIS AT THE END: scorecard summary + any failed gate w/ honest reason
  records/             # per-problem JSON records written by the runner
```

Keep modules importable both as `python -m symm.bench.run_benchmarks` (add the right path
handling) and runnable directly. Prefer stdlib `json` + plain dicts for records (no extra deps).

---

## 5. Module specifications

### 5.1 `engine.py` — jet engine (consolidate, do not rewrite)

Lift verbatim from `fisher_kpp_reduction.py`:

```python
# jet symbols: u, u_t, u_x, u_tt, u_tx, u_xx, u_ttx, u_txx, u_xxx
def Dx(f): ...          # total x-derivative on jet space
def Dt(f): ...          # total t-derivative on jet space
def prolong2(tau, xi, phi) -> dict   # keys 't','x','tt','tx','xx'
def invariance_residual(Delta, tau, xi, phi, onshell) -> sympy expr   # 0 iff symmetry
```

Add (new, for Level-0 auto-discovery — `plan.txt` Phase 1 / M1 de-risk):

```python
def determining_system(Delta, onshell, t, x, u, ansatz_degree=2):
    """Generic polynomial infinitesimal ansatz tau,xi,phi as polynomials (deg<=ansatz_degree)
    in (t,x,u) with undetermined coeffs; build pr^(2)X(Delta), substitute onshell, expand, and
    split on independent jet monomials -> list of linear determining equations + coeff symbols."""

def solve_algebra_dimension(Delta, onshell, t, x, u, ansatz_degree=2, timeout_s=1800):
    """sp.solve the determining system; return (dimension, basis_generators). MAX-RUNTIME:
    exact, full solve. Used to AUTO-RECOVER Burgers dim=5 and Fisher dim>=2 from scratch."""
```

### 5.2 `symbolic.py` — Level 0 ground truth (MAX RUNTIME)

The authority. Pure sympy, exact. Provide, per problem:

```python
SYM_CONFIG = dict(M_SYM=12, ORDER_SYM=20, SYM_CALL_CAP_S=1800, downgrade=(8,12))

def timed_sym(label, thunk):  # logs label + elapsed; enforces SYM_CALL_CAP_S; logs downgrades

def exact_solution_symbolic(problem) -> sympy expr        # the closed-form oracle, exact
def reduced_ode_symbolic(problem) -> (N_operator, vars)   # exact reduced ODE operator
def verify_reduction_symbolic(problem) -> sympy expr      # back-sub residual (must be 0)
def verify_oracle_symbolic(problem) -> sympy expr         # oracle into reduced ODE (must be 0)
def ham_series_symbolic(problem, M=SYM_CONFIG['M_SYM'], order=SYM_CONFIG['ORDER_SYM']) -> list  # u_0..u_M exact
def term_norms_symbolic(terms, hbar_val) -> list[sympy]   # exact L2 norms (or sup) per term
```

Generalize `deformation_terms` from `fisher_kpp_reduction.py` (lines 172–191) into
`ham.py:deformation_terms(Nop, Lop, eigenvalue, u0, var, M, order)` and call it here with
exact inputs. For Fisher-KPP the eigenvalue is `(k+2)(k+3)`, `Lop_s`, `u0=1-2s`, `var=s`.

### 5.3 `problems.py` — the 5 base benchmarks (uniform registry)

Each problem is a dict/dataclass with everything the harness needs. The **five base
benchmarks** (chosen to span the certificate's full behavior — see §6 for full math):

| key | PDE | reduced ODE | oracle | certificate role |
|---|---|---|---|---|
| `heat` | `u_t=u_xx` | (TW/source) linear | Gaussian/erf, exact | **R2 NON-VACUOUS** (linear ⇒ 𝓛=0): demonstrates a real bracketing bound |
| `burgers` | `u_t+u u_x=ν u_xx` | `ν U''=(U−c)U'` | Cole–Hopf tanh kink | R2 **window-contraction**: non-vacuous on small window δ<ν/k |
| `fisher` | `u_t=u_xx+u(1−u)` | `U''+cU'+U(1−U)=0` | Ablowitz–Zeppetella, c=5/√6 | **PRIMARY**; R2 **VACUOUS** (C_L·𝓛≈2≥1) ⇒ geometric-tail fallback |
| `porous` | `u_t=(u^m u_x)_x` | Barenblatt ODE | Barenblatt (m=1,2,3) | degenerate/free-boundary; residual-0 reduction; R1 interior-only |
| `blasius` | `2f'''+f f''=0` | (already ODE) | `f''(0)=0.332057...` | learned-`ħ`/shooting vs known constant |

Registry entry shape:

```python
@dataclass
class Problem:
    key: str
    title: str
    field: str                 # plan.txt §4 "Field" column
    Delta: ...                 # sympy PDE operator (None for blasius, already an ODE)
    onshell: dict              # solve Delta=0 for highest derivative
    generators: list           # (name,(tau,xi,phi)) expected symmetries, + negative controls
    reduce: callable           # PDE -> reduced ODE (returns N operator + invariant var)
    oracle: callable           # exact closed-form (sympy) + numeric lambdified
    ham_spec: dict             # bounded var, Lop, eigenvalue, u0, hbar test grid, certified band
    pinn_spec: dict            # zL,zR, BCs, collocation N, residual fn  (None if not applicable)
    expected: dict             # the §10 gate numbers for self-checking
```

### 5.4 `ham.py` — generic HAM builder

```python
def deformation_terms(Nop, Lop_inv_eigen, u0, var, M, order):
    """Generic version of fisher_kpp_reduction.deformation_terms.
    Lop_inv_eigen(k) returns the eigenvalue so L[var**k] = eigen(k)*var**k (Euler operator);
    e.g. fisher: eigen(k)=(k+2)*(k+3). Returns [u_0,...,u_M] truncated to degree<=order."""

def U_M(terms, hbar_val, upto=None):  # sum of terms (optionally first upto+1) at given hbar
```

### 5.5 `certify.py` — the three-part certificate

Implements `plan.txt §2 (S3)` exactly. Returns a structured certificate dict.

```python
def ratio_certificate(term_norms):   # R3
    """rho_k = ||u_{k+1}||/||u_k||. flag in {CERTIFIED, INCONCLUSIVE, DIVERGENT}.
    Geometric-tail bound: if rho_k<=r<1 for k>=k0:  ||U-U_M|| <= ||u_k0|| r^(M-k0+1)/(1-r).
    Mirror fisher_kpp_reduction.py lines 207-223 logic; report monotone-tail conditional flag."""

def closedness_R1(Nop, terms, ball):
    """Lemma R1: check N Lipschitz on the iterate ball (estimate local Lipschitz constant L_N
    numerically over the ball) AND that {U_M} is Cauchy (term norms summable). If both -> the
    series limit solves N[U_inf]=0. Return {lipschitz_ok, cauchy_ok, conclusion}."""

def aposteriori_R2(Nop, Lop, U_M_expr, var, domain):
    """Bound R2:  C = C_L/(1 - C_L*calligraphic_L),  ||U-U_M|| <= C*||N[U_M]||.
    Estimate C_L = ||L^{-1}|| (from eigenvalues; for Euler operator the inverse eigenvalues are
    1/((k+2)(k+3)) so C_L is bounded by max -> 1/6 for fisher), and calligraphic_L = Lipschitz
    const of the nonlinear remainder N-L over the domain.
    *** CRITICAL HONEST BRANCH ***  if C_L*calligraphic_L >= 1: bound is VACUOUS ->
    return {status:'VACUOUS', C_L, L_nl, product, fallback:'geometric-tail'} and DO NOT report a
    finite C. Otherwise return {status:'VALID', C, bound, true_error, tightness=bound/true_error}."""

def certificate(problem, terms, hbar_val, oracle):
    """Compose R3+R1+R2 into one record incl. certificate tightness and calibration inputs."""
```

For **fisher**: `C_L ≤ 1/6` (min eigenvalue (0+2)(0+3)=6), nonlinear Lipschitz `𝓛≈12` over
the unit ball of `V`, so `C_L·𝓛 ≈ 2 ≥ 1` → **VACUOUS** → must downgrade to geometric tail
(this is `plan.txt §6` risk row and §3 M2a fail-branch — it MUST be exercised and labeled).
For **heat** (linear): `𝓛=0` → `C_L·𝓛=0<1` → **VALID**, a genuine non-vacuous bound.
For **burgers** on a window `δ<ν/k`: contraction gives `C_L·𝓛<1` → **VALID**.

### 5.6 `metrics.py` — plan.txt §5 metrics + reconciliation

Implement every metric in `plan.txt §5`:

```python
def rel_L2(approx, exact, grid)            # primary accuracy
def L_inf(approx, exact, grid)
def certificate_tightness(bound, true_err) # = bound/true_error; must be >=1 to "bracket"
def certificate_calibration(flag, actually_converges)  # flag matches reality? (per problem & aggregate)
def rho_sequence_and_fit(term_norms)       # rho_k list + fitted geometric r per hbar
def equivariance_gap(pinn_model, group_action)  # target ~1e-15 (invariant ansatz) vs O(1) raw
def corrector_norm(pinn_model, U_M)        # ||N_theta||/||U_M||  (should be <<1)
def wall_clock_to_accuracy(...)            # matched-time bookkeeping
def symmetry_discovery_success(found_dim, known_dim)   # Level-0 auto-recovery
def level_agreement(level0_vals, level1_vals)          # reconciliation block (§2 rule)
```

### 5.7 `pinn.py` — Level 2 corrector (reuse the demo)

Generalize `pinn_corrector_demo.py` into a reusable function:

```python
def train_corrector(residual_fn, zL, zR, bc_values, n_coll=400, max_iter=300, seed=0):
    """Boundary-anchored: U(z)=g(z)+b(z)*net(z); g linear through BCs; b vanishes at ends.
    LBFGS strong_wolfe. RETURN model, train_time, final_residual_RMS, and a U(z) callable.
    REMEMBER: compute residual OUTSIDE torch.no_grad() (needs autograd)."""
```

Reuse the exact MLP `Linear(1,16)-Tanh-Linear(16,16)-Tanh-Linear(16,1)` (321 params) so the
Fisher number reproduces (rel-L2 8.7e-4). For the **equivariance gap** metric, the invariant
ansatz means the network sees only the invariant coordinate `z`; applying the reduction-subgroup
action to `(x,t)` must leave `u_θ` unchanged to machine precision (~1e-15). Implement that check.

### 5.8 `scorecard.py` — the ability index (§9)

### 5.9 `run_benchmarks.py` — the runner

```
for each problem in registry:
   record = {}
   # LEVEL 0 (max runtime, exact) — log progress
   record['reduce']  = verify_reduction_symbolic + invariance_residual checks + auto-discovery dim
   record['oracle']  = verify_oracle_symbolic (== 0)
   record['close']   = ham_series_symbolic (M_SYM, ORDER_SYM)
   record['certify'] = certificate(...)  (R3 + R1 + R2, honest VACUOUS handling)
   # LEVEL 1 (numeric) — fast cross-checks + most §5 metrics
   record['numeric'] = solve_ivp cross-check, minimize_scalar hbar*, numeric rho_k, rel-L2 by order
   record['level_agreement'] = reconcile(level0, level1)
   # LEVEL 2 (torch) — corrector + equivariance
   record['correct'] = train_corrector(...) (skip/NA where pinn_spec is None)
   # SCORE
   record['capability'] = per-problem capability sub-scores (§9)
   write records/<key>.json
aggregate -> scorecard (ability index) -> print + write RESULTS.md
```

---

## 6. The five benchmarks — full math (so you needn't re-derive)

### 6.1 `heat` — linear, the NON-VACUOUS R2 demonstrator
- PDE `u_t = u_xx`. Generators incl. `∂_t,∂_x`, scaling `2t∂_t+x∂_x` (note: heat HAS scaling,
  unlike Fisher). Source/similarity solution `u = t^{-1/2} exp(-x²/4t)` (exact); or TW form.
- Why included: the nonlinear remainder is **zero** (`𝓛=0`) so R2 gives `C = C_L` and a genuine,
  finite, **non-vacuous** a-posteriori bound that **brackets** the true error. This is the
  positive control proving the certificate machinery produces real bounds, not always-vacuous ones.

### 6.2 `burgers` — Cole–Hopf, the window-contraction R2
- PDE `u_t+u u_x=ν u_xx`. Five point generators (all residual **0**, from `feasibility_demo.py`):
  `∂_t`, `∂_x`, Galilean `t∂_x+∂_u`, scaling `2t∂_t+x∂_x−u∂_u`, projective
  `t²∂_t+tx∂_x+(x−tu)∂_u`. TW `z=x−ct` ⇒ `ν U''=(U−c)U'`; first integral
  `ν U'=½(U−c−k)(U−c+k)`; exact kink `U=c−k tanh(k z/2ν)` (residual **0**).
- Use `ν0=0.5, c0=1.0, k0=1.0`. On a small window `δ < ν/k` the nonlinear remainder is a
  contraction (`C_L·𝓛<1`) so R2 is **non-vacuous**; report both the window where it holds and
  that it goes vacuous on the full line. Series check: 8-term recurrence max err **1.56e-2** on
  [−1,1]; `h*` recovers `k/2ν` to **1.7e-7**.

### 6.3 `fisher` — PRIMARY, the VACUOUS-bound honest case
- PDE `u_t=u_xx+u(1−u)`. Point symmetries `∂_t,∂_x,∂_t+c∂_x` residual **0**; scaling
  `2t∂_t+x∂_x−u∂_u` is a **negative control** (residual `u(u−2)≠0` — must be reported nonzero).
- Reduce: `U''+cU'+U(1−U)=0`. Exact AZ at `c=5/√6`: `U=1/(1+B e^{z/√6})²` (residual **0**).
- HAM in bounded `s=e^{z/√6}`: `N_s[f]=s²f''+6s f'+6f(1−f)`; `L_s[f]=s²f''+6s f'+6f`
  (Euler, eigenvalue `(k+2)(k+3)`); `u0=1−2s`. Exact wave `V=1/(1+s)²` ⇒ `N_s[V]=0`.
- Certificate: ρ_k **ħ-dependent** — `ħ=−1` DIVERGENT (ρ_k crosses 1 at **1.105, 1.175**),
  `ħ=−0.6/−0.3` CERTIFIED. rel-L2 by order **1.02e-1 → 4.48e-2 → 2.04e-2** (M=3,5,7). `ħ*` by
  `minimize_scalar` in CERTIFIED band `[−0.95,−0.1]`. `solve_ivp` backward (z=6→0) matches AZ to
  **3.6e-10** (forward toward U=0 saddle is ill-conditioned — known Fisher-front property, NOT a bug).
- **R2 is VACUOUS** here: `C_L·𝓛 ≈ (1/6)(12) = 2 ≥ 1` → `certify.aposteriori_R2` must return
  `status:'VACUOUS'` and the harness must downgrade to the **geometric-tail bound** + a labeled
  *diagnostic*. This is the headline honesty result — do not paper over it.

### 6.4 `porous` — degenerate / free-boundary
- PDE `u_t=(u^m u_x)_x`, i.e. `Delta_p = u_t − (m u^{m-1} u_x² + u^m u_xx)`. Scaling symmetry
  **iff** `a=2b−m e` (verify the general residual factors and vanishes under that relation).
  Self-similar `u=t^{-α}F(η), η=x t^{-β}, α=β=1/(m+2)` ⇒ Barenblatt ODE `(F^m F')'+β(ηF)'=0`.
  Barenblatt profile `F=(C−(mβ/2)η²)_+^{1/m}` first integral `F^m F'+βηF=0` residual **0** for
  `m=1,2,3`, `β=1/(m+2)`.
- Certificate scope: R1 (closedness) holds only on the **interior** (the free boundary breaks
  Lipschitz). Report interior-only and let the ML leg own the boundary (this is `plan.txt §6`).

### 6.5 `blasius` — learned-ħ against a known constant
- ODE `2f'''+f f''=0`, BCs `f(0)=f'(0)=0, f'(∞)=1`. Oracle: `f''(0)=0.332057...`.
- Level 0: high-order Blasius series (Töpfer/Howarth scaling) to extract `f''(0)` symbolically to
  several digits; Level 1: shooting (`solve_ivp` + root-find on `f''(0)`); Level 2 optional:
  PI-OHAM-style learned correction. Gate: recover **0.332057 to 4 digits**.

---

## 7. The three-part certificate (algorithm, restated)

For each problem, after building `terms = [u_0..u_M]` at a chosen `ħ`:

**R3 — ratio / convergence (always run).**
`ρ_k = ‖u_{k+1}‖/‖u_k‖`. Flag:
- all `ρ_k<1` → `CERTIFIED` (`(cond. monotone tail)` if the ρ_k are monotone non-increasing);
- only the tail `ρ_last<1` → `CERTIFIED-conditional`;
- else → `INCONCLUSIVE / DIVERGENT`.
Geometric-tail bound when `ρ_k≤r<1` for `k≥k0`: `‖U−U_M‖ ≤ ‖u_{k0}‖·r^{M−k0+1}/(1−r)`.

**R1 — closedness lemma.** If `N` is Lipschitz on the iterate ball and `{U_M}` is Cauchy, then
`U_M→U_∞` with `N[U_∞]=0`. Implement as: numerically estimate the local Lipschitz constant of
`N` over the ball spanned by the iterates; confirm term norms are summable (Cauchy). Output the
conclusion + the scope (interior-only where there's a free boundary).

**R2 — computable a-posteriori bound.** `N=L+𝒩`; `C_L≥‖L^{-1}‖`, `𝒩` Lipschitz const `𝓛`.
If `C_L·𝓛<1`: `C=C_L/(1−C_L·𝓛)`, `‖U−U_M‖≤C·‖N[U_M]‖` → report `C`, the bound, the true error
(vs oracle), and **tightness = bound/true_error** (must be ≥1 to genuinely bracket).
If `C_L·𝓛≥1`: **VACUOUS** → report the product, refuse to print a finite `C`, fall back to R3's
geometric-tail bound, label the residual a *diagnostic*. (Heat→VALID, Burgers-window→VALID,
Fisher→VACUOUS: a deliberate trio spanning the full spectrum.)

`ħ*` selection: `argmin_ħ ‖N[U_M(·;ħ)]‖₂`, **restricted to the CERTIFIED band** so selection is
coupled honestly to the certificate (never pick a divergent ħ).

---

## 8. Metrics to compute (plan.txt §5 — implement all that apply per problem)

- **rel-L²/L∞** vs the *exact* oracle (Fisher target <1e-3 for the PINN leg).
- **certificate tightness** = (a-posteriori bound)/(true error) — must bracket (≥1) when VALID.
- **certificate calibration** = fraction of `{CERTIFIED/INCONCLUSIVE/DIVERGENT}` flags matching
  reality (aggregate across problems and ħ values).
- **ρ_k sequence + fitted geometric r** per ħ.
- **equivariance gap** (~1e-15 invariant-ansatz vs O(1) raw-coordinate) — Level 2.
- **bounded-corrector norm** `‖N_θ‖/‖U_M‖ ≪ 1` — Level 2.
- **wall-clock-to-accuracy** (matched) — record train/solve times.
- **symmetry-discovery success** = recovered algebra dim vs known (Burgers 5, Fisher ≥2) — Level 0.
- **HAM term count + symbolic-complexity growth** (where the symbolic cost explodes — the handoff order).
- **level_agreement** reconciliation (§2).

---

## 9. Capability scorecard ("ability index")

Roll per-problem results into four capability sub-scores in [0,1], then a weighted total.
The point is a **single honest number** plus a transparent breakdown — not to inflate it.

| Capability | What it measures | Per-problem pass criterion | Weight |
|---|---|---|---|
| **REDUCE** | symmetry verify + auto-discovery + residual-0 reduction | invariance residual 0 AND back-sub residual 0 AND (where attempted) auto-recovered dim correct | 0.25 |
| **CLOSE** | HAM series builds + converges for some ħ | series builds to M_SYM AND ∃ħ with all ρ_k<1 | 0.20 |
| **CERTIFY** | certificate emitted, flags calibrated, bound brackets when VALID / honestly VACUOUS when not | flag matches reality AND (R2 VALID⇒tightness≥1) OR (R2 VACUOUS⇒geometric-tail reported) | 0.35 |
| **CORRECT** | PINN accuracy + equivariance (where applicable) | rel-L2 below target AND equivariance gap ~1e-15 | 0.20 |

Rules:
- Each sub-score = (problems passing that capability) / (problems where it applies). NA problems
  excluded from the denominator (e.g. Blasius has no PDE-reduce; porous PINN may be NA).
- **Ability index** = weighted sum. Report the four sub-scores AND the total, plus a one-line
  honest caption (e.g. "CERTIFY=… reflects that Fisher's R2 is correctly VACUOUS, scored as a
  pass because the geometric-tail fallback was emitted and labeled").
- A *correctly-reported negative* (VACUOUS bound with fallback, dead generator detected, unstable
  BGI generator screened) **counts as a pass** for CERTIFY — the thesis is about honesty, not raw
  accuracy. Never score honesty as failure.
- Print a table: rows = problems, columns = the four capabilities (✓/✗/NA) + notes; then the
  weighted index. Write the same to `RESULTS.md`.

---

## 10. Verification gates (exact expected numbers — self-check against these)

The harness must reproduce these (Level 0 exact ⇒ 0; Level 1 numeric ⇒ to tolerance). Any miss
is a gate failure → log honestly in `RESULTS.md`, do not silently pass.

**REDUCE**
- Burgers: all **5** generators `invariance_residual == 0`; auto-discovery recovers **dim 5**.
- Fisher: `∂_t,∂_x,∂_t+c∂_x` residual **0**; scaling negative control residual = `u(u−2)` (nonzero).
- Fisher reduced ODE equals `U''+cU'+U(1−U)=0` (symbolic difference 0).
- Porous: scaling residual factors and vanishes under `a=2b−m e`; Barenblatt first integral **0**
  for m=1,2,3.
- AZ oracle into reduced ODE residual **0**; `N_s[1/(1+s)²]` residual **0**.

**CLOSE / CERTIFY (Fisher, primary)**
- ρ_k **ħ-dependent**: ħ=−1 DIVERGENT (crosses 1 near **1.105, 1.175**); ħ=−0.6 and ħ=−0.3 CERTIFIED.
- rel-L2 by order (M=3,5,7): **≈1.02e-1, 4.48e-2, 2.04e-2** (monotone decrease).
- `ħ*` lands inside CERTIFIED band `[−0.95,−0.1]`.
- `solve_ivp` backward cross-check vs AZ: **~3.6e-10** (≤1e-7). Forward direction ill-conditioned
  (expected, not a failure).
- R2 for Fisher: status **VACUOUS** (`C_L·𝓛≈2`); geometric-tail bound reported instead.
- R2 for Heat: status **VALID**, finite C, tightness ≥1 (brackets).

**CORRECT**
- Fisher PINN: **321 params**, train ~**0.1–0.5 s** CPU, rel-L2 **~8.7e-4** (≤1e-3), residual RMS ~3e-4.
- Equivariance gap ~**1e-15**.

**Blasius**
- `f''(0)` recovered to **0.332057** (4 digits).

**Series sanity (Burgers)**
- 8-term series max err **~1.56e-2** on [−1,1]; `h*`→`k/2ν` to **~1.7e-7**.

---

## 11. Run / verify commands

```powershell
# from D:\MATHs\math 4890 LieML
python -m symm.bench.run_benchmarks            # full run: all levels, all problems, writes records + RESULTS.md
python -m symm.bench.run_benchmarks --quick    # OPTIONAL: skip Level-0 max-runtime (M_SYM=7) for a fast smoke test
python -m symm.bench.run_benchmarks --only fisher    # single problem
```

Implement `--quick` (low fidelity, for a smoke test before the long run) and `--only <key>`.
Log progress for every Level-0 symbolic call (label + elapsed) so the 8-hour run is observable.
On completion print the scorecard table and write `symm/bench/RESULTS.md`.

Smoke-test plan: run `--quick` first; confirm Fisher numbers reproduce at low order; THEN start
the full max-runtime run.

---

## 12. Honest-negative branches (REQUIRED — these are the thesis, not edge cases)

The harness must *exercise and clearly label* every negative below. Each is a PASS for the
honesty-oriented scorecard when handled correctly:

1. **VACUOUS a-posteriori bound (Fisher):** `C_L·𝓛≥1` → no finite C; emit geometric-tail bound +
   labeled diagnostic. (`plan.txt §6` risk row; §3 M2a fail-branch.)
2. **DIVERGENT ratio (Fisher ħ=−1):** flag DIVERGENT; do not select that ħ; show calibration
   (flag matches the fact that the series actually diverges there).
3. **Dead generator / negative control (Fisher scaling):** residual nonzero `u(u−2)` →
   correctly rejected as not-a-symmetry.
4. **Ill-conditioned shooting direction (Fisher forward):** report the ~4e-2 forward error as a
   known saddle property, contrast with the well-posed backward ~3.6e-10. Not a code defect.
5. **R1 scope limit (porous free boundary):** closedness only interior; ML owns the boundary.
6. **Symbolic blow-up / timeout (Level 0):** if a call exceeds `SYM_CALL_CAP_S`, log the
   downgrade and the order at which symbolic cost explodes (the "handoff order" metric).

Never present a vacuous bound as valid; never hide a divergent flag; never average away a
level disagreement. The credibility of the whole program rests on these being visible.

---

## 13. 8-hour time budget & prioritization

Order of work so the scorecard completes even if symbolic stages run long:

1. **(0–30 min) Scaffold + smoke test.** Build `engine.py`, `problems.py`, `ham.py`, `certify.py`,
   `metrics.py`, `pinn.py`, runner skeleton. Run `--quick` and confirm Fisher low-order numbers
   reproduce (ρ_k ħ-dependence, rel-L2 trend, PINN 8.7e-4). **Checkpoint: a working low-fidelity
   scorecard exists.** Everything after this only raises fidelity.
2. **(30–90 min) Level 1 + Level 2 complete and reconciled** for all 5 problems (these are fast).
   Write per-problem records. **Checkpoint: full numeric scorecard.**
3. **(90 min–6 hr) Level 0 max-runtime.** Run the exact symbolic ground truth: full determining-
   system solve (Burgers dim 5, Fisher dim), `M_SYM=12`, `ORDER_SYM=20`, exact ratios/norms, exact
   oracle verifications. Log progress; downgrade-with-logging only on cap hits. Update records as
   each problem's Level-0 completes — so a mid-run interruption still leaves a partial but honest
   scorecard.
4. **(6–7.5 hr) Reconcile + finalize.** level_agreement blocks, certificate tightness/calibration
   aggregate, ability index. Write `RESULTS.md`.
5. **(7.5–8 hr) Buffer.** Re-run any failed gate at adjusted fidelity; document anything left at
   low fidelity. Leave the repo in a runnable, documented state.

If time runs short, **Level 0 fidelity is the thing to sacrifice** (drop to M_SYM=8/ORDER=12 with
a logged note) — never sacrifice the honest-negative labeling (§12) or the gate self-checks (§10),
which are cheap and are the actual point of the harness.

---

## 14. Acceptance checklist (the executor's definition of done)

- [ ] `symm/bench/` package builds and imports cleanly.
- [ ] `python -m symm.bench.run_benchmarks` runs end-to-end and writes `records/*.json` + `RESULTS.md`.
- [ ] All §10 gate numbers reproduced (or each miss logged with an honest reason).
- [ ] Level 0 = pure sympy, exact, max-runtime, with progress logging and logged downgrades.
- [ ] Level 1 ↔ Level 0 reconciliation (`level_agreement`) present for every problem.
- [ ] Three-part certificate emitted per problem; Fisher R2 correctly **VACUOUS** + geometric-tail
      fallback; Heat R2 **VALID** + bracketing; calibration computed.
- [ ] All §12 honest-negatives exercised and labeled.
- [ ] Capability scorecard + single ability index printed and in `RESULTS.md`, with honest caption.
- [ ] `README.md` explains how to run and what each metric means.

— end of build plan —
