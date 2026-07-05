# LieML — Project Advisory

*2026-07-03. A whole-project assessment at professional mathematical scope, written against the verified in-tree state (40-test suite green as of today). Part I is the advisory; Part II is the requested brief guide to operating the machine-learning side. Every number in this document is reproduced by a runnable script named next to it.*

---

## 0. Executive summary

1. **The thesis spine is sound and mostly built.** REDUCE → CLOSE → CERTIFY → CORRECT is a coherent, falsifiable program, and the riskiest software — the symmetry engine — exists, is exact (rational arithmetic), resumable, fuzz-tested, and matches textbook answers where checked (Burgers dim 5; porous medium m=2 dim 4 = the generic branch of the classical group classification). What is still missing is *integration*: no single command yet runs a PDE end-to-end through all four stages, and the draft article has no results section.
2. **Positioning discipline is the biggest publication risk.** Computing Lie symmetry algebras is mature computer-algebra territory (Maple `rifsimp`/DifferentialAlgebra, CRACK/LIEPDE, GeM, SYM). Your defensible claims are the *reducibility record* (a persistent, comparable object that also stores barren "LATENT" cases), the *certified pipeline composition*, and the *verified-prioritizer reductor* — never "a new symmetry algorithm". §2.4 gives the exact claim/prior-art table.
3. **Today's measured ML results (honest protocol):** the duplicate-structure audit deflated the old baselines (1-NN on dim: 0.685 → 0.591 once test PDEs are genuinely unseen structures); the interaction-learning tree then **beats 1-NN decisively on dim (0.747 vs 0.591; fin-only 0.832 vs 0.707)** — the "learned reductor" gate is passed — **but fails the high-symmetry safety gate as a dim-predictor (recall 0.000 on dim ≥ 4)**. Training the *triage* target directly (binary dim ≥ 4) fixes it: acc 0.892, recall 0.792, precision 0.559. The triage tree is the usable prioritizer.
4. **The audit caught a real mathematical phenomenon, not noise:** one structure, `u_t = u + 2u³ + c·x·uₓ`, gives a stable 2-dimensional algebra at c = −1/2 but a growing sequence 2, 3, 4, 7, 13 at c = 2 (verified at degree 5). That is a coefficient *resonance* — exactly the "special value" locus that group classification is about, and exactly what presence-only features cannot see. §3 explains why this makes "engine always verifies" mathematically forced rather than a policy choice.
5. **Best next move: enumerate, don't sample.** The order-≤2 dictionary has only 2¹¹ − 1 = 2047 possible term-structures, and the engine averages ~0.26 s/PDE — complete ground truth for the whole current universe costs ~9 CPU-minutes. That converts the reductor evaluation from sampling statistics into an exact train/test experiment and yields the full jump-candidate list for free.
6. **Two bars, clearly separated:** the course-thesis bar is reachable with the textbook validation table plus one end-to-end run (gates G1 + G5 in §5). The external-publication bar additionally needs the enumeration study, the group-classification demo, and citation re-verification (G2–G4, G7).

---

# PART I — ADVISORY

## 1. Where the project stands (verified assets only)

| Asset | State | Evidence |
|---|---|---|
| Symmetry engine v1 (`symm/sym_engine/`, 12 modules) | Complete | 40-test suite; Burgers → dim 5 CERTIFIED via CLI; resume bit-identical |
| Scope of engine | Scalar (1+1) evolution PDEs, polynomial data, **point** symmetries, polynomial infinitesimal ansatz of degree D iterated to stabilization | spec §5–6; Deviation Note |
| Reducibility record | dim (exact), algebra type (derived/lower-central series), defect δ, index ρ, reproducibility tuple π, status CERTIFIED/PARTIAL/**LATENT-ONLY** | `reducibility.py`; spec §6 |
| PDE generator + fuzz gate | 12-term dictionary; every random PDE's discovered generators satisfy residual ≡ 0, exact dim = numeric SVD nullity | `test_generator_fuzz.py` |
| Corpus v1 | 200 PDEs, 0 errors, seed 0, presence features + stalled labels | `symm/reductor/corpus_v1.json` |
| Reductor v1.1 | Evaluation harness + interaction tree; results in §0.3 and Part II | `symm/reductor/train_reductor.py` (pinned by `test_train_reductor.py`) |
| CCS stages CLOSE/CERTIFY/CORRECT | Exist as verified standalone demos (Fisher HAM in the bounded variable, three-part certificate incl. the deliberately vacuous branch, 321-param invariant PINN) | `symm/fisher_kpp_reduction.py`, `symm/pinn_corrector_demo.py` |
| SAC-NO | Three reproducing probes + plan | `symm/sacno/` |
| Benchmark harness | Specified, **not built** | `symm/bench/BUILD_PLAN.md` |
| Documents | spec (PDF), plan, draft article (no results yet), intro (22 pp), zero-to-one guide | `docs/`, `draft/`, `intro.tex` |

Known accepted-for-v1 gaps (ledger): no uniqueness guard in `_init_onshell`; no explicit semisimple flag; structure constants not in the record signature; per-degree resume deferred.

## 2. The mathematical landscape you are standing in

*(First principles, briefly — each paragraph names where to read more, preferring files already in `ref/`.)*

**2.1 Classical symmetry analysis.** A point symmetry is a vector field X = τ∂ₜ + ξ∂ₓ + φ∂ᵤ whose prolongation annihilates the PDE on its solution set. The crucial structural fact: even for a *nonlinear* PDE, the determining equations for (τ, ξ, φ) are **linear** — the prolongation formula is linear in the infinitesimals — so "find all symmetries" is linear algebra over function spaces, which is why your determining-matrix + nullspace design is the right shape. Canonical sources: Olver (`ref/Olver_LieGroups_DiffEq_g.pdf`), the lecture notes in `ref/Notes on Lie symmetry group methods…`, Kruglikov–Lychagin (`ref/KruglikovLychagin_GeometryDEs.pdf`), and, for the finiteness/Tresse side, `ref/LieTresse_2011.05101.pdf`. **Group classification** — how the algebra jumps as coefficients/parameters vary — goes back to Ovsiannikov; the porous-medium equation u_t = (uᵐuₓ)ₓ with its special exponent m = −4/3 is the standard example, and your engine already reproduces its generic branch (dim 4 at m = 2).

**2.2 The computer-algebra lineage.** Complete algorithmic treatments of determining systems are differential elimination: Rosenfeld–Gröbner (Boulier et al.; Maple `DifferentialAlgebra`) and Reid's rif algorithm (`rifsimp`), with mature packages CRACK/LIEPDE (Wolf, REDUCE), GeM (Cheviakov, Maple), SYM (Dimas–Tsoubelis, Mathematica). These return the *full* solution structure, including non-polynomial infinitesimals and arbitrary-function (infinite-dimensional) parts. Your degree-iterated elimination is a deliberate deviation (recorded in the spec's Deviation Note): **dim_D is a certified lower bound within the polynomial window; stabilization is evidence, not proof, of completeness; the `stalled` flag marks suspected infinite parts; full completeness is delegated to the Maple escape hatch.** Keep that sentence, verbatim if you like, everywhere a dim is reported — it is the difference between an honest tool and an overclaim.

**2.3 The ML-for-PDE stack and its critique.** PINNs (Raissi et al. 2019; practical failure modes in `ref/An Expert's Guide to Training Physics-informed Neural Networks…`), neural operators (`ref/FOURIER NEURAL OPERATOR…`), and the critique line your validation policy is built on: `ref/Weak baselines and reporting biases…` (McGreivy–Hakim) plus the two review/synthesis PDFs in `ref/`. The five Lie+ML papers you extracted (`ref/Brandstetter_LPSDA_2202.07643.pdf`, `ref/Helwig_GFNO_2306.05697.pdf`, `ref/Lie Algebra Canonicalization…`, `ref/NeurIPS-2023-lie-point-symmetry-and-physics-informed-networks…`, `ref/Generalized Lie Symmetries in Physics-Informed Neural Operators.pdf`, plus `ref/LieSolver…`) all use symmetry as an *inductive bias* (penalty, augmentation, equivariant architecture, canonicalization) and all hit the documented bottlenecks (dead training signal at symmetric points, non-compact-group cost, no certificates, symmetry derivation done by hand). Your program uses symmetry as *coordinates/reduction* — that is the seam you occupy; the draft's §2 taxonomy already says this correctly.

**2.4 Novelty audit — claim vs. nearest named prior art.**

| Your claim | Nearest prior art | Verdict |
|---|---|---|
| Automated Lie-algebra computation | `rifsimp`, CRACK/LIEPDE, GeM, SYM | **Not novel.** Position as: open, exact, resumable, *record-producing* pure-sympy pipeline with fuzz-tested invariants — an engineering/reproducibility contribution |
| Reducibility record ρ, δ, π + **LATENT** storage of "unsolvable" PDEs | Handbook classification tables (Ibragimov); integrability detectors (Painlevé tests) | **Plausibly novel as a database/record concept.** δ must always be labeled a π-dependent diagnostic, never an invariant |
| Learned reductor as *verified prioritizer* for classification sweeps | ML-guided symbolic math (Lample–Charton); no named system in the Lie-symmetry niche | **Likely novel niche.** Value is contingent on a demonstrated sweep speedup (gate G4) |
| Certified pipeline (reduce + HAM + ratio/a-posteriori certificate + corrector), vacuous branch reported as a feature | Turkyilmazoglu's ratio criterion; classical a-posteriori estimation; none composed with symmetry reduction + ML correction | **The thesis-level novelty.** Keep the vacuous-on-Fisher branch front and center — falsifiability is the selling point |
| SAC-NO: equivariance by coordinates, pathology *structurally* absent | LPSDA (augmentation), equivariant/canonicalized operators | Defensible as a structural argument; needs the parametric-operator comparison as evidence |

## 3. The mathematical spine: semicontinuity, and why your design is forced

Fix a term-structure (which dictionary terms appear with nonzero coefficient) and let the coefficient vector c vary. Every entry of the determining matrix A(c) is polynomial in c, so **rank is lower-semicontinuous in c: the nullity — the symmetry dimension — is constant on a Zariski-open (generic) set and can only jump UP on special algebraic subvarieties of coefficient space.** This single classical fact organizes everything you have measured:

- *Why coefficient-valued features hurt the classifier* (your v1.1 negative result): generically, dim does not depend on c at all, so coefficient magnitudes are pure noise to a distance model.
- *Why presence features are the right generic representation, and also provably insufficient*: the upward jumps live on measure-zero subvarieties that presence flags cannot express. Your audit found one at rate 1/187 structures: `u_t = u + 2u³ + c·x·uₓ` has windowed dims [2, 2] at c = −1/2 but [2, 3, 4, 7, 13] at c = 2 (replayed and verified at D ≤ 5; corpus rows 96/132). Both equations are first-order, hence integrable by characteristics, and classical Lie theory supplies infinite-dimensional *contact* symmetry families for first-order equations; whether the *point* algebra is infinite is equation-dependent, and what your polynomial window sees depends on a **resonance**: the characteristic flow scales like x·e^(−ct), and polynomial symmetry solutions exist in quantity only when the relevant exponents combine integrally — c = 2 is resonant, c = −1/2 is not.
- *Why "prioritizer, not filter" + "engine always verifies" is forced, not policy*: a structure-based model is at best generically right; the interesting objects of group classification are precisely the special subvarieties it cannot see. The reductor may order the search; only the engine may assert.
- *What group classification IS in this language*: mapping the jump locus. Your enumeration proposal (G2) is a complete jump-locus survey of a small dictionary — a legitimate, self-contained mathematical table.

One caution for the write-up: the corpus label `dim` is the *windowed* dimension at Dmax with the stall flag; the resonance example shows the windowed label itself is not coefficient-invariant. So state findings as "windowed-dim is structure-determined for 186/187 sampled structures; the exception is a verified resonance", not "dim is structure-determined".

## 4. Referee-style risk list

- **R1 — Completeness of the polynomial window.** A referee's first question. Mitigation: the lower-bound sentence of §2.2 everywhere; the heat equation as the canonical demonstration that the `stalled` flag correctly detects the infinite β∂ᵤ part; `rifsimp` escape hatch for any headline claim.
- **R2 — Scope (scalar, 1+1, evolution).** Fine for a thesis; say it in the abstract, not in a footnote. Note the cost drivers for extensions: ansatz size p = 3·C(D+3, 3) grows polynomially, but systems and extra spatial dimensions multiply the jet combinatorics.
- **R3 — δ is not an invariant.** Already handled by the π-tuple; never compare δ across different π.
- **R4 — The current corpus universe is exhaustible.** 2047 structures ≈ 9 CPU-minutes: a referee could enumerate it and call the learned model a lookup table. Pre-empt by doing the enumeration yourself (G2) and repositioning the reductor as the tool for regimes where enumeration dies (larger dictionaries, parametric coefficients, systems).
- **R5 — Label noise at resonances / stall-window artifacts.** Quantified: 1/187. Keep quantifying it; it is a feature of the story (it *is* the classification phenomenon), but only if reported.
- **R6 — Pipeline integration debt.** CLOSE/CERTIFY/CORRECT currently consume hand-prepared reductions, not engine output. Until G5 exists, the four-stage pipeline is a diagram, not a result.
- **R7 — Citation hygiene.** Everything cited from `ref/` is verified by reading; the classical attributions (Ovsiannikov, Bluman–Kumei, Ibragimov, Boulier, Reid, Wolf, Cheviakov, Raissi, Lample–Charton) are standard but their exact bibliographic data must be re-verified against the actual papers before any external submission, per your standing rule.
- **R8 — Triage precision is 0.559,** i.e. ~44% of flagged PDEs are false alarms. For a *prioritizer* this is acceptable by cost asymmetry — a false alarm wastes seconds of engine time, a missed rich PDE loses a discovery — but say so explicitly rather than hiding the number.

## 5. Prioritized roadmap with falsifiable gates

| Gate | Task | PASS criterion | Cost |
|---|---|---|---|
| **G1** | **Textbook validation table**: heat (6 ⊕ ∞), Burgers (5), Fisher (2), porous m=2 (4), KdV u_t + uuₓ + uₓₓₓ (4), transport | Engine dim matches the published table, or the discrepancy is explained by the window and confirmed via escape hatch | hours |
| **G2** | **Enumerate, don't sample (v1.2)**: all 2047 order-≤2 structures, one generic draw each + a second draw to flag jump candidates | Complete table; fuzz invariants hold on all; jump-candidate list published with the resonance of §3 as entry #1 | ~9 CPU-min compute + a day of writing |
| **G3** | **Reductor on complete ground truth**: train/test split *by structure* over the G2 table | Beats 1-NN on dim and majority on triage; triage recall ≥ 0.8 reported with precision; 3 extracted rules verified symbolically | a day |
| **G4** | **Group-classification demo**: sweep u_t = (uᵐuₓ)ₓ, engine detects the m = −4/3 jump; measure sweep time with vs. without reductor prioritization | Jump found automatically; honest speedup number (even if modest) | days |
| **G5** | **One-command end-to-end CCS-SYM+ run on Fisher**: engine → reduction → HAM + certificate (existing) → PINN corrector (existing), validated only against the exact similarity solution | Single script; certificate numbers reproduced; vacuous branch reported as such | days |
| **G6** | Build the benchmark harness per `symm/bench/BUILD_PLAN.md` | Its own acceptance tests | days |
| **G7** | Draft: results section from G1–G5, claims tightened per §2.4, citations re-verified | Compiles; every claim traceable to a script | days |

**Status (2026-07-03): G1 COMPLETE — 6/6 PASS, review clean.** `symm/textbook_validation.py` + `symm/tests/test_textbook_validation.py` (suite 45 pass). Four finite dims match the literature (Burgers 5, Fisher 2, porous m=2 → 4, KdV → 4), each corroborated by exact-rational-dim = numeric-SVD-nullity; heat and transport are correctly flagged infinite by the `stalled`+growth signature (no false finite number). Opus review independently re-confirmed all six literature values. G2–G7 not started.

**Two bars.** Course thesis = G1 + G5 (+ the already-done engine and reductor story, negatives included). External publication (e.g. a symbolic-computation or SciML venue) = additionally G2–G4 + G7. Do them in the numbered order; G1 is first because it is cheap and every later claim leans on it.

---

# PART II — BRIEF GUIDE TO OPERATING THE MACHINE LEARNING

*(Everything runs on this machine: CPU-only torch 2.12, numpy, no sklearn — models are hand-rolled on purpose so every number is inspectable. All commands from the project root.)*

### II.1 What "ML" means in this project — three distinct uses

1. **The reductor** (live): a *tabular* classifier over 12 binary presence flags → predicts symmetry richness. Cheap, interpretable, verified by the engine. This is the one you operate below.
2. **The PINN corrector** (`symm/pinn_corrector_demo.py`): a 321-parameter network fitting the *residual* of the HAM series on the reduced ODE, in invariant coordinates.
3. **SAC-NO** (`symm/sacno/`): operator learning on the orbit quotient.

### II.2 The learning problem, in one box

> **X** ∈ {0,1}¹² — which dictionary terms appear (presence flags).
> **y** — either `dim` (windowed symmetry dimension), or `klass` (inf/fin), or `triage` = [dim ≥ 4].
> **Data** — engine self-labels; no human annotation. `symm/reductor/corpus_v1.json`, 200 rows, seed 0.
> **Judge** — leave-one-out cross-validation (LOOCV): hold out one example, fit on the rest, predict it, repeat; the accuracy is an honest estimate of "how well does this generalize" without spending data on a separate test set (right choice at n ≈ 200).

### II.3 Five rules (each one is a lesson this project already paid for)

1. **Baselines first.** Majority-class and 1-nearest-neighbor cost nothing. A model that does not beat them has learned nothing, whatever its loss curve says.
2. **Determinism.** Fixed seeds, no hidden RNG (the evaluation harness has none at all), tie-breaks specified. Same command ⇒ same number, forever.
3. **No leakage — split by structure, not by row.** Our own case study: 200 rows contain only 187 distinct structures; with duplicates in the neighbor pool, 1-NN on dim scored 0.685, but on genuinely unseen structures it is **0.591**. The inflated number was *real code, honestly computed, wrong protocol.* This is the single most common way ML results lie.
4. **Pre-register the headline.** We fixed depth 3 *before* looking, and print the whole depth curve so selection optimism is visible (depth 6 scores higher — that number is shopping, not science).
5. **A negative, correctly labeled, is a result.** Coefficient features *hurt* (0.685 → 0.395): that measurement became the semicontinuity insight of Part I §3. Log negatives in `.sdd/progress.md` like everything else.

### II.4 Operate it

```bash
# 1. Rebuild the corpus from scratch (~1 min; deterministic, seed 0)
python -c "from symm.reductor.generator import build_corpus; import json, pathlib; pathlib.Path('symm/reductor/corpus_v1.json').write_text(json.dumps(build_corpus(200, seed=0), indent=1), encoding='utf-8')"

# 2. Run the full honest evaluation (audit + baselines + tree + triage + rules)
python symm/reductor/train_reductor.py

# 3. Regression-lock: the whole suite, including the pinned scoreboard numbers
python -m pytest symm/tests -q        # expect: 40 passed
```

### II.5 Current scoreboard (2026-07-03, struct-LOOCV = honest protocol)

| Target | majority | 1-NN | tree d=3 | verdict |
|---|---|---|---|---|
| klass (inf/fin), n=186 | **0.898** | 0.871 | 0.849 | saturated by majority — not the interesting target |
| dim (all) | 0.543 | 0.591 | **0.747** | tree wins: interactions are real signal |
| dim (fin only) | 0.605 | 0.707 | **0.832** | tree wins |
| dim-predictor recall on dim ≥ 4 | — | 0.833 | 0.000 | **tree FAILS the safety gate** (buries every rich PDE) |
| triage = [dim ≥ 4] | 0.871 | acc 0.839 / prec 0.435 | **acc 0.892 / rec 0.792 / prec 0.559** | train the triage target directly — this is the prioritizer |

Row-LOOCV (duplicate-inflated, kept only for continuity with the ledger): 1-NN 0.920 / 0.685 / 0.757.

### II.6 Read the model — rules are conjectures

The depth-3 triage tree is human-readable: *if none of {u²uₓₓ, uₓ², uuₓₓ} is present, predict high symmetry.* Mathematically plausible — those are the strongly nonlinear couplings that destroy symmetry — but treat every extracted rule as a **conjecture to hand back to the engine** (exhaustively checkable once G2 exists). That loop — model proposes, engine certifies — is the project's ML philosophy in one line.

### II.7 Overfitting, in five sentences

A model with enough capacity can memorize 186 rows perfectly and know nothing; accuracy on *training* data is therefore meaningless. Capacity is controlled here by the depth cap (a depth-3 tree over binary features can express at most 8 rules). LOOCV simulates deployment on unseen data; splitting by structure makes "unseen" true. The depth curve shows the temptation: deeper trees score better *here* partly by luck, and picking the best depth after looking is itself a small leak (hence the pre-registered d=3). When in doubt, prefer the simpler model and report both.

### II.8 When you graduate to torch (PINN / SAC-NO)

Same rules, two additions. **(a)** Never judge by the training loss: validate against the exact similarity solution (the McGreivy–Hakim discipline — closed-form ground truth or nothing). **(b)** On this CPU, stay small (hundreds of parameters, seconds of training — the existing demos are sized correctly); if a run needs minutes, first ask whether the *problem* got bigger or the *model* got vainer. Start from `symm/pinn_corrector_demo.py` and `symm/sacno/sacno_parametric_operator.py`, which already follow both rules.

### II.9 The supervisor hook

`symm/sym_engine/interfaces.py::review()` is the DeepSeek-V4-Pro supervisor stub; it activates only if `DEEPSEEK_API_KEY` is set in the environment (never hardcode the key; it is never written to checkpoints, reports, or guides — there is a test asserting this). The supervisor reviews; it never overrides an engine verdict.

---

*Files this advisory rests on: `symm/reductor/train_reductor.py` (all Part II numbers), `symm/tests/test_train_reductor.py` (pins), `.sdd/progress.md` (ledger), `docs/superpowers/specs/2026-06-26-lie-symmetry-engine-v1-design.pdf` (defect/index definitions), `draft/draft.tex` (positioning). Suite: 40 passed.*
