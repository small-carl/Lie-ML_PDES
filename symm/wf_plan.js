export const meta = {
  name: 'lieml-research-plan',
  description: 'Literature sweep + gap synthesis + judge-panel novelty selection + methodology + feasibility computation for a Lie-symmetry / semi-analytical / ML PDE research plan',
  phases: [
    { title: 'Recon', detail: 'parallel multi-angle literature sweep (web + local refs)' },
    { title: 'Synthesize', detail: 'dedup findings into a gap map + candidate theses' },
    { title: 'Theses', detail: 'generate 4 distinct candidate research theses' },
    { title: 'Judge', detail: '3 lenses score each thesis adversarially' },
    { title: 'Methodology', detail: 'design methodology, benchmark suite, validation' },
    { title: 'Feasibility', detail: 'run sympy Lie-symmetry reduction on a benchmark PDE' },
    { title: 'Assemble', detail: 'synthesize the full phased plan' },
  ],
}

const GOAL = "Semi-Analytical and Machine Learning Methods for Solving PDEs Using Lie Symmetry Analysis. Reduce complex PDEs into simpler forms and obtain exact or approximate analytical solutions, with applications in mathematical physics, chemistry, and engineering.";

const WEB = "FIRST load web tools: call ToolSearch with query 'select:WebSearch,WebFetch'. Then run 4-7 WebSearch queries and WebFetch the 2-4 most important sources. Prioritize arXiv, peer-reviewed journals, and recent work (2020-2026). For every claim, capture a concrete citation (title, authors, year, venue/arXiv id, URL). Be skeptical: note what is actually demonstrated vs. asserted.";

const LOCAL = "Local reference markdown lives under D:/MATHs/math 4890 LieML/pdf2md/output/<NAME>/document.md . Relevant NAMEs: 'Generalized Lie Symmetries in Physics-Informed Neural Operators', 'NeurIPS-2023-lie-point-symmetry-and-physics-informed-networks-Paper-Conference', 'LIE_ALGEBRA_CANONICALIZATION equivariant-neural-operators-under-arbitrary-lie-groups-Paper-Conference', 'FOURIER NEURAL OPERATOR FOR parametric partial differential equations'. Read those relevant to your angle.";

const LIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['angle','sota_summary','key_papers','methods_techniques','gaps_open_problems','promising_directions'],
  properties: {
    angle: { type: 'string' },
    sota_summary: { type: 'string', description: '200-400 word grounded summary of the state of the art for this angle' },
    key_papers: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['title','authors_year','venue_or_id','key_contribution','relevance'],
      properties: {
        title: { type: 'string' }, authors_year: { type: 'string' },
        venue_or_id: { type: 'string', description: 'journal/conf or arXiv id + URL' },
        key_contribution: { type: 'string' }, relevance: { type: 'string' } } } },
    methods_techniques: { type: 'array', items: { type: 'string' } },
    gaps_open_problems: { type: 'array', items: { type: 'string' } },
    promising_directions: { type: 'array', items: { type: 'string' } },
  },
};

const GAP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['consolidated_sota','gap_map','candidate_thesis_seeds','strongest_benchmarks'],
  properties: {
    consolidated_sota: { type: 'string' },
    gap_map: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['gap','why_it_matters','evidence'],
      properties: { gap: { type: 'string' }, why_it_matters: { type: 'string' }, evidence: { type: 'string' } } } },
    candidate_thesis_seeds: { type: 'array', items: { type: 'string' } },
    strongest_benchmarks: { type: 'array', items: { type: 'string', description: 'benchmark PDEs with rich symmetry + known exact/similarity solutions, suitable for limited-compute validation' } },
  },
};

const THESIS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['title','core_idea','why_novel','methodology_sketch','benchmark_pdes','limited_compute_feasibility','risks'],
  properties: {
    title: { type: 'string' }, core_idea: { type: 'string' },
    why_novel: { type: 'string', description: 'explicit contrast with named prior art' },
    methodology_sketch: { type: 'string' },
    benchmark_pdes: { type: 'array', items: { type: 'string' } },
    limited_compute_feasibility: { type: 'string' },
    risks: { type: 'array', items: { type: 'string' } },
  },
};

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens','overall','strengths','weaknesses','verdict'],
  properties: {
    lens: { type: 'string' },
    overall: { type: 'number', description: '1-10 overall quality of this thesis on this lens' },
    strengths: { type: 'array', items: { type: 'string' } },
    weaknesses: { type: 'array', items: { type: 'string' } },
    verdict: { type: 'string' },
  },
};

const METHOD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['refined_thesis','pipeline_stages','math_formulation','benchmark_suite','baselines','metrics','tooling','validation_strategy','novel_claims','risks_mitigations'],
  properties: {
    refined_thesis: { type: 'string' },
    pipeline_stages: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['stage','what','how'], properties: { stage:{type:'string'}, what:{type:'string'}, how:{type:'string'} } } },
    math_formulation: { type: 'string', description: 'the core equations: invariance condition, reduction, semi-analytical series, ML loss/architecture, with LaTeX' },
    benchmark_suite: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['pde','field','symmetry_note','why'], properties: { pde:{type:'string'}, field:{type:'string'}, symmetry_note:{type:'string'}, why:{type:'string'} } } },
    baselines: { type: 'array', items: { type: 'string' } },
    metrics: { type: 'array', items: { type: 'string' } },
    tooling: { type: 'array', items: { type: 'string', description: 'concrete software: sympy, SADE/GeM/Maple PDEtools, DeepXDE, PyTorch, etc.' } },
    validation_strategy: { type: 'string' },
    novel_claims: { type: 'array', items: { type: 'string' } },
    risks_mitigations: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['risk','mitigation'], properties: { risk:{type:'string'}, mitigation:{type:'string'} } } },
  },
};

const COMPUTE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pde','approach','code','outputs','generators_found','reduced_equation','reduced_solution','verification','what_it_demonstrates'],
  properties: {
    pde: { type: 'string' }, approach: { type: 'string' },
    code: { type: 'string', description: 'the actual runnable python/sympy code used' },
    outputs: { type: 'string', description: 'verbatim key outputs printed by the code' },
    generators_found: { type: 'array', items: { type: 'string' } },
    reduced_equation: { type: 'string' },
    reduced_solution: { type: 'string' },
    verification: { type: 'string', description: 'how correctness was checked (substitution residual, etc.)' },
    what_it_demonstrates: { type: 'string' },
  },
};

// Phase 1: Recon
phase('Recon')
const ANGLES = [
  { key:'classical-lie', focus:'Classical Lie symmetry analysis of PDEs: infinitesimal generators, prolongation, determining equations, optimal systems, group classification, and SOFTWARE (Maple PDEtools/Lie, GeM, SADE, SYM, DESOLV, DimSym, Python libraries). Recent reviews and algorithmic advances 2018-2026.' },
  { key:'reduction-types', focus:'Symmetry reductions and solution construction: similarity/invariant solutions; NONCLASSICAL (conditional) symmetries; potential and nonlocal symmetries; APPROXIMATE symmetries (Baikov-Gazizov-Ibragimov) for perturbed PDEs; conservation laws. Which give exact solutions classical Lie misses.' },
  { key:'semi-analytical', focus:'Semi-analytical / series methods for PDEs and ODEs: Homotopy Analysis Method (HAM) and its convergence-control parameter h, Adomian Decomposition (ADM), Variational Iteration (VIM), Differential Transform (DTM), Homotopy Perturbation (HPM), Residual Power Series. Convergence theory, error, and limitations.' },
  { key:'lie-plus-semianalytical', focus:'Existing COMBINATIONS of Lie symmetry reduction WITH semi-analytical methods (reduce PDE to ODE via similarity variable, then apply HAM/ADM/DTM/VIM). Survey what has been published, on which equations, and what is missing/weak.' },
  { key:'symmetry-informed-ml', focus:'Symmetry-informed / equivariant ML for PDEs: symmetry-loss PINNs, generalized/evolutionary-representative symmetry losses, symmetry-enhanced PINNs via invariant surface conditions, equivariant neural operators (FNO/DeepONet), Lie algebra canonicalization, symmetry-informed Gaussian processes. Strengths and documented failure modes (e.g. point symmetries giving no training signal).' },
  { key:'ml-discovers-symmetry', focus:'ML/automation that DISCOVERS symmetries or governing equations: LieGAN and symmetry-discovery networks, learning Lie point symmetries from data, neural/automated determining-equation solving, plus equation discovery (SINDy, PDE-FIND, symbolic regression) and how symmetry constrains discovery.' },
  { key:'applications-benchmarks', focus:'Application PDEs with rich Lie symmetry and known exact/similarity solutions across math physics, CHEMISTRY, and engineering: Burgers, KdV/mKdV, Fisher-KPP and reaction-diffusion, nonlinear heat / porous-medium, Fokker-Planck, Black-Scholes, boundary-layer/Blasius, chemical kinetics and combustion, fractional and variable-coefficient PDEs. Which are best canonical benchmarks for limited-compute validation.' },
  { key:'gaps-critiques', focus:'Critiques, open problems, and review/survey articles (2022-2026) on symmetry + ML + semi-analytical PDE solving: error/convergence guarantees, high dimensionality, stiffness/multiscale, inverse problems, reproducibility, where hybrid analytical-ML methods genuinely beat pure ML or pure numerics. Find the honest open questions.' },
]
const recon = (await parallel(ANGLES.map(a => () => agent(
  'You are a research-literature analyst. Research angle "' + a.key + '" for this study:\n\nGOAL: ' + GOAL + '\n\nFOCUS: ' + a.focus + '\n\n' + WEB + '\n\n' + LOCAL + '\n\nReturn rigorous, citation-grounded findings. Distinguish what is proven from what is claimed. Be specific about gaps that a new study could exploit.',
  { label:'lit:' + a.key, phase:'Recon', schema: LIT_SCHEMA }
)))).filter(Boolean)

// Phase 2: Synthesize
phase('Synthesize')
const gaps = await agent(
  'You are the research synthesizer. Below are ' + recon.length + ' literature analyses across distinct angles for this study.\n\nGOAL: ' + GOAL + '\n\nFINDINGS (JSON):\n' + JSON.stringify(recon) + '\n\nProduce: (1) a consolidated, honest state-of-the-art; (2) a GAP MAP of genuine, exploitable gaps at the intersection of Lie symmetry + semi-analytical methods + ML (favor gaps that are heuristically NEW and feasible with limited compute); (3) candidate thesis seeds; (4) the strongest benchmark PDEs (rich symmetry + known exact/similarity solutions) for limited-compute validation. Avoid restating well-trodden symmetry-loss-PINN work as if novel.',
  { label:'gap-synthesis', phase:'Synthesize', schema: GAP_SCHEMA }
)

// Phase 3: Candidate theses
phase('Theses')
const THESIS_ANGLES = [
  { key:'reduce-then-semianalytic-ml', framing:'REDUCTION-FIRST HYBRID: use Lie (and where useful nonclassical/conditional) symmetry to reduce the PDE to an ODE/lower-D PDE; solve the reduced equation semi-analytically (e.g. HAM/RPS) yielding an interpretable parametric series with convergence-control parameters; use ML to (a) optimally select convergence-control parameters, (b) learn a residual correction, (c) extend the local series globally and enforce boundary/initial data; symmetry invariance acts as a hard inductive bias throughout. Emphasize interpretability + error control.' },
  { key:'approximate-symmetry-perturbed', framing:'APPROXIMATE / NONCLASSICAL SYMMETRY for REAL (perturbed, near-symmetric) PDEs: many physical/chemical/engineering PDEs only have symmetries approximately (small dissipation, weak nonlinearity, variable coefficients). Combine Baikov-Gazizov-Ibragimov approximate symmetry theory + perturbation/homotopy semi-analytical solution of the reduced problem, with ML handling the non-reducible remainder. This targets a real gap: symmetry-ML work almost always assumes exact symmetry.' },
  { key:'ml-discover-reduce-solve', framing:'CLOSED DISCOVERY LOOP: ML/automated symbolic methods DISCOVER the (possibly data-driven or hidden) Lie/nonclassical symmetries of a given PDE, then trigger AUTOMATED similarity reduction, then a semi-analytical closed-form/series solution of the reduced equation, with automatic verification by residual substitution. Targets equations where hand-derivation of symmetries is hard, and unifies discovery+reduction+solution+verification into one pipeline.' },
  { key:'similarity-coordinate-operator', framing:'SYMMETRY-COORDINATE / SIMILARITY-ANCHORED NEURAL SOLVERS: build the neural ansatz/operator directly in similarity (invariant) coordinates so the architecture is equivariant by construction, and ANCHOR/curriculum-train it on the semi-analytical similarity solutions of the reduced equation; scale from single-solution PINN to a parametric neural operator that respects the symmetry group. Targets the documented "no training signal" weakness of point-symmetry losses by building symmetry into coordinates+data rather than loss.' },
]
const candidates = (await parallel(THESIS_ANGLES.map(t => () => agent(
  'You are a PDE+ML research-thesis architect. Develop ONE rigorous, heuristically-NEW research thesis with this framing:\n\n' + t.framing + '\n\nGOAL: ' + GOAL + '\n\nGAP ANALYSIS (JSON):\n' + JSON.stringify(gaps) + '\n\nGround it against named prior art (cite from the gap analysis). Make the novelty crisp and the limited-compute feasibility concrete. State what exactly would be demonstrated on a benchmark with sympy + a small PINN/HAM, not a GPU cluster.',
  { label:'thesis:' + t.key, phase:'Theses', schema: THESIS_SCHEMA }
)))).filter(Boolean)

// Phase 4: Judge
phase('Judge')
const LENSES = [
  { key:'novelty', prompt:'NOVELTY vs prior art: is this genuinely new, or a relabel of existing symmetry-loss-PINN / standard symmetry-reduction work? Penalize incremental framings. Reward defensible, citable novelty.' },
  { key:'rigor', prompt:'MATHEMATICAL RIGOR and CORRECTNESS: are the symmetry, reduction, semi-analytical convergence, and ML claims mathematically sound and verifiable? Flag hand-waving.' },
  { key:'feasibility', prompt:'FEASIBILITY UNDER LIMITED COMPUTE: can the core be demonstrated with sympy + a small PINN/HAM on a laptop CPU, no large GPU? Reward clean, falsifiable, limited-compute milestones.' },
]
const judged = await parallel(candidates.map((c, i) => () =>
  parallel(LENSES.map(l => () => agent(
    'You are an adversarial reviewer applying the ' + l.key.toUpperCase() + ' lens. ' + l.prompt + '\n\nGOAL: ' + GOAL + '\n\nTHESIS (JSON):\n' + JSON.stringify(c) + "\n\nScore 'overall' 1-10 on this lens (be discriminating; do not cluster at 7-8). Justify.",
    { label:'judge:t' + i + ':' + l.key, phase:'Judge', schema: JUDGE_SCHEMA }
  ))).then(votes => ({ candidate: c, votes: votes.filter(Boolean) }))
))
const scored = judged.map(j => {
  const os = j.votes.map(v => v.overall).filter(x => typeof x === 'number')
  const avg = os.length ? os.reduce((a,b)=>a+b,0)/os.length : 0
  return { candidate: j.candidate, avg, votes: j.votes }
}).sort((a,b) => b.avg - a.avg)
const winner = scored[0]
const runnersUp = scored.slice(1)
log('Top thesis: "' + (winner && winner.candidate && winner.candidate.title) + '" (avg ' + (winner && winner.avg.toFixed(2)) + '). Runners-up: ' + runnersUp.map(r=>r.avg.toFixed(2)).join(', '))

// Phase 5: Methodology
phase('Methodology')
const methodology = await agent(
  'You are the principal methodologist. Turn the WINNING thesis into a rigorous methodology, GRAFTING the best ideas from the runners-up where they strengthen it.\n\nGOAL: ' + GOAL + '\n\nWINNER (JSON):\n' + JSON.stringify(winner.candidate) + '\n\nWINNER JUDGE FEEDBACK (address the weaknesses):\n' + JSON.stringify(winner.votes) + '\n\nRUNNER-UP THESES to graft from (JSON):\n' + JSON.stringify(runnersUp.map(r=>r.candidate)) + '\n\nGAP ANALYSIS + BENCHMARKS (JSON):\n' + JSON.stringify(gaps) + '\n\nGive the core math (invariance condition pr^{(k)}X(Delta)=0 on Delta=0, reduction via invariants, the semi-analytical series with convergence-control parameter, the ML loss/architecture), a concrete benchmark suite spanning physics/chemistry/engineering, baselines, metrics, real tooling, and a falsifiable validation strategy with limited-compute milestones.',
  { label:'methodology', phase:'Methodology', schema: METHOD_SCHEMA }
)

// Phase 6: Feasibility
phase('Feasibility')
const feasibility = await agent(
  'You are a computational mathematician. Produce a SMALL, REAL, RUNNABLE feasibility computation that demonstrates the core reduction step of the methodology on ONE canonical benchmark PDE chosen from: ' + JSON.stringify((gaps && gaps.strongest_benchmarks) || ['Burgers equation','nonlinear heat/porous-medium equation','Fisher-KPP equation']) + '.\n\nGOAL: ' + GOAL + '\nMETHODOLOGY (JSON):\n' + JSON.stringify(methodology) + '\n\nYou have a real shell. Run python (sympy/numpy/scipy ARE installed: python 3.14, sympy 1.14, numpy, scipy). Use the Bash or PowerShell tool to actually execute code in a temp file under D:/MATHs/math 4890 LieML/symm/ . Do this concretely:\n1) Take a known Lie point symmetry generator of the chosen PDE and VERIFY it satisfies the linearized symmetry (invariance) condition via prolongation in sympy (show the residual is 0 on solutions).\n2) Construct the similarity variable and reduce the PDE to an ODE; verify the reduction by substitution.\n3) Solve or approximate the reduced ODE (sympy dsolve, OR a few-term HAM/series, OR a tiny numeric/PINN sketch).\nReport the ACTUAL printed outputs verbatim. If a step fails, fix it and rerun. Keep total runtime modest. Return the runnable code and real outputs.',
  { label:'feasibility-compute', phase:'Feasibility', schema: COMPUTE_SCHEMA }
)

// Phase 7: Assemble
phase('Assemble')
const draftPlan = await agent(
  'You are the lead author. Write a rigorous, well-structured research plan in GitHub-flavored Markdown for this study, integrating everything below. It must read as a heuristically-new, defensible, limited-compute-feasible plan.\n\nGOAL: ' + GOAL + '\n\nCHOSEN THESIS (JSON):\n' + JSON.stringify(winner.candidate) + '\n\nMETHODOLOGY (JSON):\n' + JSON.stringify(methodology) + '\n\nFEASIBILITY COMPUTATION (JSON):\n' + JSON.stringify(feasibility) + '\n\nCONSOLIDATED GAPS + KEY CITATIONS (JSON):\n' + JSON.stringify(gaps) + '\n\nStructure: (0) one-paragraph thesis + novelty statement; (1) Background and positioning vs named prior art; (2) Core methodology with the key math; (3) Phased plan (Phase 0..N) — each phase with objective, concrete tasks, tools, deliverable, and a falsifiable success criterion; (4) Benchmark suite table (physics/chemistry/engineering); (5) The worked feasibility example (with the real computed generators/reduction); (6) Evaluation: baselines + metrics; (7) Risks and mitigations; (8) Expected contributions / publishable claims; (9) A compact end-to-end workflow diagram in text/pseudocode. Be specific and cite. Do NOT invent results beyond the feasibility computation provided.',
  { label:'assemble-plan', phase:'Assemble' }
)

return {
  ranking: scored.map(s => ({ title: s.candidate && s.candidate.title, avg: s.avg })),
  winner: winner.candidate,
  gaps,
  methodology,
  feasibility,
  draftPlan,
}
