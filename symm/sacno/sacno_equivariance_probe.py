# ============================================================================
# SAC-NO  PROBE 1 -- EQUIVARIANCE-BY-COORDINATES (the headline structural claim)
# ----------------------------------------------------------------------------
# SAC-NO = Similarity-Anchored, Coordinate-equivariant Neural Operator, the
# companion thesis to CCS-SYM+.  Its headline is that symmetry equivariance is
# achieved by the CHOICE OF INPUT COORDINATES, not by a soft loss penalty:
# the operator reads the orbit-quotient (Lie-invariant) coordinates, so the
# documented "no training signal at a symmetric point" pathology of
# loss-penalty approaches (Generalized-Lie-PINO, arXiv:2502.00373) is
# STRUCTURALLY ABSENT -- the equivariance error is exactly zero by construction,
# independent of training, seed, or data.
#
# This probe demonstrates that on the heat/porous SCALING group
#       g_lambda : (x, t, u)  ->  (lambda x, lambda^2 t, u)        (u invariant)
# whose orbit-quotient is parameterised by the single similarity invariant
#       z(x, t) = x / sqrt(t).
# A network that reads z is EXACTLY invariant under g_lambda (the group is
# divided out before the net sees anything); a network that reads raw (x, t) is
# generically NOT, with an O(1) equivariance gap that no amount of architecture
# tweaking removes (only a loss can push it down -- and only where it is trained).
#
# We report the EQUIVARIANCE GAP
#       E(N) = RMS_{points, group elements} | N(g.p) - N(p) |  /  RMS | N(p) |
# for both nets, untrained, at random init.  Double precision throughout so the
# invariant net's gap reports the true floating-point floor (~1e-16), not a
# single-precision artefact.
# ============================================================================
import torch, math

torch.set_default_dtype(torch.float64)          # so the "machine-eps" floor is real
torch.manual_seed(0)

# ---- the same small MLP body used everywhere in this project (CPU, tiny) -----
def mlp(in_dim, width=16):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 1))

# Two networks with IDENTICAL bodies; they differ only in what they are fed.
#   raw net  N_raw(x, t)         -- sees the ambient coordinates
#   inv net  N_inv(z), z = x/sqrt(t)  -- sees the orbit-quotient coordinate
net_raw = mlp(2)
net_inv = mlp(1)

# ---- the group action g_lambda and the invariant chart -----------------------
def group_action(x, t, lam):
    """Heat/porous scaling symmetry: (x,t) -> (lam*x, lam^2*t); u unchanged."""
    return lam * x, (lam ** 2) * t

def invariant_coord(x, t):
    """The Lie invariant of g_lambda: z = x / sqrt(t)  (constant along orbits)."""
    return x / torch.sqrt(t)

def N_raw(x, t):
    return net_raw(torch.cat([x, t], dim=1))

def N_inv(x, t):
    z = invariant_coord(x, t)
    return net_inv(z)

# ---- sample interior points (t>0) and a grid of group elements ---------------
M = 4000
x = torch.empty(M, 1).uniform_(-3.0, 3.0)
t = torch.empty(M, 1).uniform_(0.25, 4.0)          # strictly positive -> z well defined
lambdas = torch.tensor([0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]).reshape(-1)

def equivariance_gap(N):
    """RMS over (points x group elements) of |N(g.p) - N(p)|, normalised by RMS|N(p)|."""
    with torch.no_grad():
        base = N(x, t)
        num_sq, den_sq, count = 0.0, 0.0, 0
        for lam in lambdas:
            xg, tg = group_action(x, t, lam)
            diff = N(xg, tg) - base
            num_sq += (diff ** 2).sum().item()
            den_sq += (base ** 2).sum().item()
            count += base.numel()
        return math.sqrt(num_sq / count) / math.sqrt(den_sq / count)

gap_raw = equivariance_gap(N_raw)
gap_inv = equivariance_gap(N_inv)

# sanity: confirm the invariant coordinate really is constant along an orbit
with torch.no_grad():
    z0 = invariant_coord(x, t)
    drift = 0.0
    for lam in lambdas:
        xg, tg = group_action(x, t, lam)
        drift = max(drift, (invariant_coord(xg, tg) - z0).abs().max().item())

print("=" * 70)
print("SAC-NO PROBE 1 -- EQUIVARIANCE GAP (untrained nets, double precision)")
print("=" * 70)
print(f"group: g_lambda (x,t,u) -> (lambda x, lambda^2 t, u);  invariant z = x/sqrt(t)")
print(f"points sampled: {M:,}   group elements: {list(lambdas.numpy())}")
print(f"net body (both): Linear(.,16)-Tanh-Linear(16,16)-Tanh-Linear(16,1)")
print("-" * 70)
print(f"max drift of z along orbits           : {drift:.3e}   (must be ~machine eps)")
print(f"equivariance gap, INVARIANT-coord net : {gap_inv:.3e}   <-- exact by construction")
print(f"equivariance gap, RAW (x,t)      net  : {gap_raw:.3e}   <-- O(1), structural")
print(f"ratio raw/invariant                   : {gap_raw / max(gap_inv,1e-300):.3e}")
print("-" * 70)
ok = (gap_inv < 1e-12) and (gap_raw > 1e-2)
print(f"VERDICT: invariant net structurally equivariant (gap<1e-12) "
      f"and raw net is not (gap>1e-2): {'PASS' if ok else 'FAIL'}")
print("Interpretation: SAC-NO divides the symmetry group out in the COORDINATE")
print("map, so equivariance error is zero before training begins -- there is no")
print("'no-gradient-at-symmetric-points' failure mode for a loss to fight.")
