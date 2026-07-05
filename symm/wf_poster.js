export const meta = {
  name: 'lieml-poster-and-review',
  description: 'Refresh the June-2026 PDE-solver method landscape, adversarially review intro.tex, and synthesize poster-ready comparison material for the CCS-SYM+ proposal',
  phases: [
    { title: 'Landscape', detail: 'web refresh of popular PDE-solving methods up to June 2026 + adversarial review of intro.tex' },
    { title: 'Synthesize', detail: 'poster-ready comparison table + positioning vs current methods' },
  ],
}

const GOAL = "Semi-Analytical and Machine-Learning Methods for Solving PDEs Using Lie Symmetry Analysis. The proposed method is CCS-SYM+: Lie-symmetry REDUCE -> homotopy-analysis CLOSE -> theorem-backed convergence/error CERTIFY -> small symmetry-invariant ML CORRECT, validated on closed-form exact similarity solutions on a laptop CPU.";

const WEB = "FIRST load web tools: call ToolSearch with query 'select:WebSearch,WebFetch'. Run 5-8 WebSearch queries; WebFetch 2-4 key sources. The current month is June 2026 - prioritise 2024-2026 work. Capture concrete citations (title, authors, year, venue/arXiv id, URL). Distinguish demonstrated from claimed.";

const LANDSCAPE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['methods','newest_2026_trends','honest_caveats'],
  properties: {
    methods: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['name','category','how_it_works','strengths','weaknesses_2026','popularity','key_refs'],
      properties: {
        name: { type: 'string' },
        category: { type: 'string', description: 'classical-numerical | physics-informed | neural-operator | generative | symmetry/geometric | semi-analytical | foundation-model | hybrid' },
        how_it_works: { type: 'string' },
        strengths: { type: 'array', items: { type: 'string' } },
        weaknesses_2026: { type: 'array', items: { type: 'string' } },
        popularity: { type: 'string', description: 'how popular/active as of June 2026, with evidence' },
        key_refs: { type: 'array', items: { type: 'string' } } } } },
    newest_2026_trends: { type: 'array', items: { type: 'string', description: 'newest 2025-2026 directions (PDE foundation models, transformer neural operators, diffusion/generative solvers, KANs, etc.) with a citation' } },
    honest_caveats: { type: 'array', items: { type: 'string', description: 'documented critiques: weak baselines, FEM-beats-PINN, generalization, cost' } },
  },
};

const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['overall_correctness_1to10','errors','pedagogy_suggestions','verdict'],
  properties: {
    overall_correctness_1to10: { type: 'number' },
    errors: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['location','issue','severity','fix'],
      properties: { location:{type:'string'}, issue:{type:'string'}, severity:{type:'string', description:'critical|major|minor'}, fix:{type:'string'} } } },
    pedagogy_suggestions: { type: 'array', items: { type: 'string' } },
    verdict: { type: 'string' },
  },
};

const POSTER_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['tagline','positioning_paragraph','comparison_rows','our_pillars','elevator_claims','where_we_win','where_we_dont'],
  properties: {
    tagline: { type: 'string', description: 'one punchy line for the poster banner' },
    positioning_paragraph: { type: 'string' },
    comparison_rows: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['family','representative','core_idea','strengths','limitations','contrast_with_ccs_sym'],
      properties: { family:{type:'string'}, representative:{type:'string'}, core_idea:{type:'string'},
        strengths:{type:'string'}, limitations:{type:'string'}, contrast_with_ccs_sym:{type:'string'} } } },
    our_pillars: { type: 'array', items: { type: 'string', description: 'the 3-5 pillars/contributions of CCS-SYM+' } },
    elevator_claims: { type: 'array', items: { type: 'string' } },
    where_we_win: { type: 'array', items: { type: 'string' } },
    where_we_dont: { type: 'array', items: { type: 'string', description: 'honest scope limits vs popular methods' } },
  },
};

// Phase 1: landscape (2 web agents) + intro review (1 agent), in parallel
phase('Landscape')
const tasks = [
  () => agent(
    'You are a survey analyst. Map the CURRENT (up to June 2026) landscape of POPULAR methods for solving PDEs, broadly: classical numerics (FEM, finite differences, spectral), physics-informed NNs (PINNs and variants), neural operators (FNO, DeepONet, PINO, transformer/Galerkin operators, graph operators), GENERATIVE/diffusion PDE solvers, geometric/equivariant & symmetry-informed methods, semi-analytical series methods (HAM/ADM/VIM), and emerging PDE FOUNDATION MODELS. For each, give how it works, strengths, June-2026 weaknesses, popularity with evidence, and refs.\n\n' + WEB + '\n\nContext (our method): ' + GOAL,
    { label:'landscape:broad', phase:'Landscape', schema: LANDSCAPE_SCHEMA }
  ),
  () => agent(
    'You are a trend scout. Find the NEWEST 2025-2026 directions in ML-for-PDEs that a June-2026 proposal poster must acknowledge to look current: PDE/scientific FOUNDATION MODELS and large pretrained operators, TRANSFORMER-based neural operators, DIFFUSION/flow-matching & generative PDE solvers, Kolmogorov-Arnold Networks (KANs) for PDEs, operator-learning generalization/scaling results, and the latest honest BENCHMARK/critique papers (e.g. McGreivy-Hakim follow-ups, PDEBench/PINNacle updates). Concrete citations required.\n\n' + WEB + '\n\nContext (our method): ' + GOAL,
    { label:'landscape:newest', phase:'Landscape', schema: LANDSCAPE_SCHEMA }
  ),
  () => agent(
    'You are an adversarial mathematics + pedagogy reviewer. READ the file D:/MATHs/math 4890 LieML/intro.tex (use the Read tool). It is an introductory article on Lie symmetry of PDEs + ML for a near-beginner. Check EVERY mathematical and historical claim for correctness, especially: the generator/algebra/exponential-map story; the prolongation invariance condition; the claim that the viscous Burgers point algebra is 5-dimensional and the linear heat equation is 6-dimensional + infinite-dimensional superposition part; the similarity-reduction "variable compression" claims (Burgers tanh, porous-medium Barenblatt with beta=1/(m+2), Fisher-KPP non-integrable TW); the optimal-system/conjugacy classification; the distinction between Lie groups, finite-DIMENSIONAL Lie algebras, and FINITE GROUPS OF LIE TYPE (Chevalley groups); and the compact-vs-noncompact remark about equivariant ML. Flag any error with a precise fix. Also give pedagogy suggestions. Be exacting; do not rubber-stamp.',
    { label:'review:intro', phase:'Landscape', schema: REVIEW_SCHEMA }
  ),
]
const out = await parallel(tasks)
const landscape = [out[0], out[1]].filter(Boolean)
const review = out[2]

// Phase 2: synthesize poster-ready material
phase('Synthesize')
const poster = await agent(
  'You are the proposal lead. Using the June-2026 method landscape below, produce POSTER-READY comparison material that positions our method (CCS-SYM+) crisply and HONESTLY against the popular families. Do not over-claim; include where we do NOT compete (e.g. high-D, full-field operator learning, non-invariant data).\n\nOUR METHOD: ' + GOAL + '\n\nLANDSCAPE (JSON):\n' + JSON.stringify(landscape) + '\n\nProduce a tagline, a positioning paragraph, comparison rows (one per popular family: classical numerics, PINNs, neural operators, generative/diffusion, symmetry-informed ML, plain semi-analytical, foundation models), our pillars, elevator claims, where-we-win, and where-we-dont.',
  { label:'poster-synthesis', phase:'Synthesize', schema: POSTER_SCHEMA }
)

return { landscape, review, poster }
