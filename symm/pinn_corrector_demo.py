# Stage-S4 verification: a SMALL CPU PINN solves the Lie-reduced Fisher-KPP BVP.
# Reduced ODE (travelling wave, c=5/sqrt6):  U'' + c U' + U(1-U) = 0,
# exact Ablowitz-Zeppetella front  U_ex(z) = 1/(1 + B e^{z/sqrt6})^2  (B=1).
# Boundary-anchored ansatz so BCs are exact:  U(z) = g(z) + b(z)*N_theta(z),
# with g linear through the two BC values and b(z) vanishing at the endpoints.
# Goal: confirm the ML leg is laptop-CPU-cheap and accurate vs the EXACT oracle.
import torch, time, math
torch.manual_seed(0)
c  = 5.0/math.sqrt(6.0); s6 = math.sqrt(6.0); B = 1.0
zL, zR = -8.0, 8.0
def U_ex(z): return 1.0/(1.0 + B*torch.exp(z/s6))**2
uL, uR = U_ex(torch.tensor(zL)).item(), U_ex(torch.tensor(zR)).item()

net = torch.nn.Sequential(torch.nn.Linear(1,16), torch.nn.Tanh(),
                          torch.nn.Linear(16,16), torch.nn.Tanh(),
                          torch.nn.Linear(16,1))
nparams = sum(p.numel() for p in net.parameters())
def g(z):  return uL + (uR-uL)*(z-zL)/(zR-zL)          # satisfies the two BCs
def b(z):  return (z-zL)*(zR-z)/((zR-zL)**2)           # vanishes at both ends
def U(z):  return g(z) + b(z)*net(z)

zc = torch.linspace(zL, zR, 400).reshape(-1,1).requires_grad_(True)
def residual():
    u  = U(zc)
    uz = torch.autograd.grad(u, zc, torch.ones_like(u), create_graph=True)[0]
    uzz= torch.autograd.grad(uz, zc, torch.ones_like(uz), create_graph=True)[0]
    return uzz + c*uz + u*(1.0-u)

opt = torch.optim.LBFGS(net.parameters(), max_iter=300, line_search_fn='strong_wolfe')
t0 = time.time()
def closure():
    opt.zero_grad(); R = residual(); loss = (R**2).mean(); loss.backward(); return loss
opt.step(closure)
train_s = time.time()-t0

final_res = (residual().detach()**2).mean().sqrt().item()
with torch.no_grad():
    zg = torch.linspace(zL, zR, 801).reshape(-1,1)
    err = (U(zg) - U_ex(zg)).abs()
    relL2 = (err.pow(2).mean().sqrt() / U_ex(zg).pow(2).mean().sqrt()).item()
print(f"params={nparams}  train_time={train_s:.2f}s  final_residual_RMS={final_res:.2e}")
print(f"max|U_pinn - U_exact| on [{zL},{zR}] = {err.max().item():.3e}")
print(f"relative L2 error vs exact Ablowitz-Zeppetella front = {relL2:.3e}")
