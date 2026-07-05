# From Zero to One: the LieML Symmetry Engine

## 1. The math in five lines
A symmetry is a generator X = tau d_t + xi d_x + phi d_u. Prolong it to derivatives, demand
pr^(k)X(Delta)=0 on solutions, and split on the free jet monomials -> the determining system. Its
solution space is the symmetry algebra. Pick a generator, find its invariants, substitute -> the PDE
collapses to an ODE. See ../../intro.tex sections 2-4 for the why.

## 2. Architecture (one line each)
- jet.py         generic prolongation + the PDE type
- determining.py the determining matrix A from a degree-D polynomial ansatz
- eliminate.py   degree-iterated exact-rational elimination -> the algebra (with checkpoints)
- algebra.py     generators, dimension, structure constants, algebra type
- reduce.py      characteristic invariants -> residual-zero reduced ODE
- reducibility.py exact dim (invariant) + metric defect/index (diagnostic)
- checkpoint.py  srepr<->JSON save/load + step log
- crosscheck.py  independent numerical SVD nullity
- engine.py      the state machine: run(resume=True)
- report.py / interfaces.py  human report + reductor/supervisor hooks

## 3. Run it
    python -m pytest symm/tests -q
    python -m symm.sym_engine_cli run burgers --workdir symm/runs/burgers --no-resume

## 4. What you get back (the reducibility record)
dim (exact, invariant) | algebra type | reductions (residual 0) | defect delta |
index rho in [0,1] | params pi | status. LATENT-ONLY = "no symmetry, saved anyway".
delta and rho are defined precisely in
docs/superpowers/specs/2026-06-26-lie-symmetry-engine-v1-design.tex (section 6).

## 5. Resume
The engine checkpoints each degree to symm/runs/<name>/checkpoint-NNN.json and saves the final
record to result.json. Re-running the same command with resume returns the saved record; a fresh run
is deterministic, so an interrupted run reproduces the same result.

## 6. Worked trace (Burgers)
- degree D=1: nullity=4
- degree D=2: nullity=5
- degree D=3: nullity=5
- STATUS=CERTIFIED dim=5 delta=0.000e+00 index=1.000

## 7. How it will grow
propose_reductions() (interfaces.py) will become a learned ranker; review() will call
DeepSeek-V4-Pro with the key read from the DEEPSEEK_API_KEY environment variable (never hardcoded).
Both interfaces already exist as deterministic stubs.
