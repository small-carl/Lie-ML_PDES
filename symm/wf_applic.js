export const meta = {
  name: 'lieml-review-and-applicability',
  description: 'Adversarially re-review intro.tex and deep-search the web to evaluate the real-world applicability of the project methods (Lie reduction, HAM, convergence certificate, symmetry-invariant ML)',
  phases: [
    { title: 'Assess', detail: 'review intro.tex + parallel web applicability deep-dives per method/domain' },
    { title: 'Synthesize', detail: 'applicability scorecard + honest verdict' },
  ],
}

const PROJECT = "Project CCS-SYM+: solve PDEs by (1) Lie/nonclassical/approximate-symmetry REDUCTION to a low-dim ODE, (2) CLOSE it with a homotopy-analysis (HAM) semi-analytical series, (3) CERTIFY with a runtime convergence+error certificate on the reduced equation, (4) CORRECT with a small symmetry-invariant CPU PINN; validated against closed-form exact similarity solutions. Application domains: mathematical physics, chemistry, engineering.";

const WEB = "FIRST load web tools: ToolSearch query 'select:WebSearch,WebFetch'. Do a DEEP search: 6-9 WebSearch queries and WebFetch 3-5 key sources. Current month June 2026; prioritise 2020-2026. Every claim needs a concrete citation (title, authors, year, venue/arXiv id, URL). Be adversarial about applicability: separate what is genuinely deployed/validated from what is only demonstrated on toy problems.";

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

const APPLIC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['component','applicability_summary','where_it_applies','where_it_fails','maturity_evidence','real_case_studies','applicability_score_1to10','honest_verdict'],
  properties: {
    component: { type: 'string' },
    applicability_summary: { type: 'string' },
    where_it_applies: { type: 'array', items: { type: 'string' } },
    where_it_fails: { type: 'array', items: { type: 'string' } },
    maturity_evidence: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['claim','citation'], properties: { claim:{type:'string'}, citation:{type:'string'} } } },
    real_case_studies: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['domain','example','outcome','citation'],
      properties: { domain:{type:'string'}, example:{type:'string'}, outcome:{type:'string'}, citation:{type:'string'} } } },
    applicability_score_1to10: { type: 'number' },
    honest_verdict: { type: 'string' },
  },
};

const ASSESS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['executive_summary','scorecard','domain_fit','overall_verdict','recommendations','risks_to_applicability'],
  properties: {
    executive_summary: { type: 'string' },
    scorecard: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['component','applicability','maturity','key_limit','score'],
      properties: { component:{type:'string'}, applicability:{type:'string'}, maturity:{type:'string'}, key_limit:{type:'string'}, score:{type:'number'} } } },
    domain_fit: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['domain','fit','examples'], properties: { domain:{type:'string'}, fit:{type:'string'}, examples:{type:'string'} } } },
    overall_verdict: { type: 'string' },
    recommendations: { type: 'array', items: { type: 'string' } },
    risks_to_applicability: { type: 'array', items: { type: 'string' } },
  },
};

phase('Assess')
const jobs = [
  () => agent(
    'You are an adversarial mathematics + pedagogy reviewer. READ D:/MATHs/math 4890 LieML/intro.tex (Read tool). Check EVERY mathematical and historical claim, with extra scrutiny on the NEW section 2.3 on matrix (linear) Lie groups (Hall): that a matrix Lie group is a closed subgroup of GL(n,C); that so(n)=skew-symmetric, su(n)=skew-Hermitian, sl(n)=traceless; the matrix-exponential / one-parameter-subgroup statement; the Baker-Campbell-Hausdorff formula X+Y+1/2[X,Y]+...; and the representation/equivariance intertwining claim rho_out(g) Phi = Phi o rho_in(g). Also re-verify earlier claims: Burgers 5-dim and heat 6-dim+infinite point algebras; Barenblatt beta=1/(m+2); Fisher-KPP non-integrable TW; the Lie-group vs finite-dimensional-algebra vs finite-groups-of-Lie-type distinction. Flag each error with a precise fix and severity. Do not rubber-stamp.',
    { label:'review:intro', phase:'Assess', schema: REVIEW_SCHEMA }
  ),
  () => agent(
    'Evaluate the REAL-WORLD APPLICABILITY of Lie / nonclassical / approximate (BGI) SYMMETRY REDUCTION of PDEs. How many real PDEs across mathematical physics, chemistry, and engineering actually admit useful symmetry reductions vs. how many do not (variable coefficients, complex geometry, boundary/initial conditions that break symmetry, high dimension)? Cover software maturity (Maple PDEtools/GeM/SADE, SymPy) and concrete published case studies. ' + WEB + '\n\nCONTEXT: ' + PROJECT,
    { label:'applic:lie-reduction', phase:'Assess', schema: APPLIC_SCHEMA }
  ),
  () => agent(
    'Evaluate the REAL-WORLD APPLICABILITY of SEMI-ANALYTICAL series methods (HAM/OHAM, ADM, VIM, DTM) and especially the reduce-by-similarity-then-HAM pattern. Where are they genuinely used and validated (boundary layers, nanofluids, heat transfer, fractional PDEs, finance), and where do they fail (convergence not guaranteed, far-field/strong-nonlinearity breakdown, hand-tuned auxiliary parameters, the "questionable papers" critique)? ' + WEB + '\n\nCONTEXT: ' + PROJECT,
    { label:'applic:semi-analytical', phase:'Assess', schema: APPLIC_SCHEMA }
  ),
  () => agent(
    'Evaluate the REAL-WORLD APPLICABILITY of (a) symmetry-informed/equivariant ML for PDEs and small ML CORRECTORS on analytic priors, and (b) whether per-problem CONVERGENCE/ERROR CERTIFICATES (a-posteriori error estimation) are valued and adopted in scientific ML and numerical analysis. Is a runtime certificate a real selling point, and who needs it (verification, safety-critical, certified computation)? What is the adoption reality of symmetry-informed PINNs/operators? ' + WEB + '\n\nCONTEXT: ' + PROJECT,
    { label:'applic:ml-and-certificate', phase:'Assess', schema: APPLIC_SCHEMA }
  ),
  () => agent(
    'Evaluate APPLICATION-DOMAIN FIT for the project methods, with concrete published case studies: mathematical physics (Burgers/KdV/porous-medium/Fokker-Planck), CHEMISTRY (reaction-diffusion, combustion, chemical kinetics, Belousov-Zhabotinsky), and ENGINEERING (boundary-layer/Blasius, transport in porous media, option pricing/Black-Scholes via Lie symmetry). For each domain state where symmetry-reduction + semi-analytical/ML realistically applies and where it does not. ' + WEB + '\n\nCONTEXT: ' + PROJECT,
    { label:'applic:domains', phase:'Assess', schema: APPLIC_SCHEMA }
  ),
]
const res = await parallel(jobs)
const review = res[0]
const applic = [res[1], res[2], res[3], res[4]].filter(Boolean)

phase('Synthesize')
const assessment = await agent(
  'You are the principal investigator writing an HONEST applicability assessment of the project methods. Integrate the per-component web evaluations below into a scorecard, a domain-fit table, an overall verdict (is this method genuinely applicable, and to what fraction of real problems?), concrete recommendations to maximise applicability, and the main risks to applicability. Do NOT inflate; state plainly where the method is niche.\n\nPROJECT: ' + PROJECT + '\n\nPER-COMPONENT APPLICABILITY (JSON):\n' + JSON.stringify(applic),
  { label:'applicability-assessment', phase:'Synthesize', schema: ASSESS_SCHEMA }
)

return { review, applic, assessment }
