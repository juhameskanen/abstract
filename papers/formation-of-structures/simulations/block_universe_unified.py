import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# =============================
# CONFIG
# =============================
T = 350
H = 256
W = 236
N_OBS = 3
ITER = 300

kappa = 0.001      # spectral weight
alpha = 0.05       # geometric MDL weight
beta = 2.0         # coupling weight

h_start = 0.2
h_end = 0.95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# HELPERS
# =============================

def build_k2(T, H, W):
    kt = torch.fft.fftfreq(T).reshape(T,1,1).repeat(1,H,W)
    ky = torch.fft.fftfreq(H).reshape(1,H,1).repeat(T,1,W)
    kx = torch.fft.fftfreq(W).reshape(1,1,W).repeat(T,H,1)
    return (kx**2 + ky**2 + kt**2).to(device)

def entropy(p):
    p = torch.clamp(p, 1e-12)
    return -torch.sum(p * torch.log(p))

def mdl_loss(traj):
    loss = 0.0
    for t in range(2, traj.shape[0]):
        pred = traj[t-1] + (traj[t-1] - traj[t-2])
        loss += torch.sum((traj[t] - pred)**2)
    return loss

def bilinear_sample(field, pos):
    """
    field: (H, W)
    pos: (2,) in normalized [0,1]
    """
    x = pos[0] * (W - 1)
    y = pos[1] * (H - 1)

    x0 = torch.clamp(x.long(), 0, W-2)
    y0 = torch.clamp(y.long(), 0, H-2)

    dx = x - x0.float()
    dy = y - y0.float()

    f00 = field[y0, x0]
    f10 = field[y0, x0+1]
    f01 = field[y0+1, x0]
    f11 = field[y0+1, x0+1]

    return (f00*(1-dx)*(1-dy) +
            f10*dx*(1-dy) +
            f01*(1-dx)*dy +
            f11*dx*dy)

# =============================
# INITIALIZE VARIABLES
# =============================

k2 = build_k2(T,H,W)

# psi = 0.01 * torch.randn((T,H,W,2), device=device, requires_grad=True)
psi = torch.randn((T,H,W,2), device=device).detach().requires_grad_(True)


# seed low entropy
psi.data[0, H//2, W//2, 0] = 1.0
psi.data *= 0.01

# observer trajectories (normalized coords)
# obs = torch.rand((N_OBS, T, 2), device=device, requires_grad=True)
obs = torch.rand((N_OBS, T, 2), device=device).detach().requires_grad_(True)

print(psi.is_leaf)  # should be True
print(obs.is_leaf)  # should be True

optimizer = torch.optim.Adam([psi, obs], lr=0.01)

h_max = np.log(H*W)

# =============================
# TRAIN
# =============================

for it in range(ITER):
    optimizer.zero_grad()

    psi_c = torch.complex(psi[...,0], psi[...,1])

    # --- spectral loss ---
    psi_k = torch.fft.fftn(psi_c, dim=(0,1,2))
    spec_loss = torch.sum(k2 * torch.abs(psi_k)**2)

    entropy_loss = 0.0
    coupling_loss = 0.0

    for t in range(T):
        prob = torch.abs(psi_c[t])**2
        p_norm = prob / (torch.sum(prob) + 1e-12)

        # entropy constraint
        progress = t / (T-1)
        target_h = (h_start + (h_end-h_start)*progress) * h_max
        h = entropy(p_norm)

        entropy_loss += (h - target_h)**2

        # coupling: observers must sit in high-density regions
        for i in range(N_OBS):
            pos = obs[i,t]
            density = bilinear_sample(prob, pos)

            # target: nonzero stable amplitude
            coupling_loss += (density - 0.02)**2

    # --- geometric MDL ---
    geom_loss = 0.0
    for i in range(N_OBS):
        geom_loss += mdl_loss(obs[i])

    # total loss
    loss = (
        kappa * spec_loss +
        entropy_loss +
        alpha * geom_loss +
        beta * coupling_loss
    )

    loss.backward()
    optimizer.step()

    # keep observers inside bounds
    with torch.no_grad():
        obs.clamp_(0.0, 1.0)

    if it % 20 == 0:
        print(f"Iter {it} | Loss {loss.item():.4f}")

# =============================
# RENDER
# =============================

psi_c = torch.complex(psi[...,0], psi[...,1])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("unified_sim.mp4", fourcc, 20, (W, H))

for t in range(T):
    prob = torch.abs(psi_c[t])**2
    frame = prob / (prob.max() + 1e-12)
    # frame = (frame.cpu().numpy() * 255).astype(np.uint8)
    frame = frame.detach().cpu().numpy()
    frame = (frame * 255).astype(np.uint8)

    frame = cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)

    # draw observers
    for i in range(N_OBS):
        x = int(obs[i,t,0].item() * (W-1))
        y = int(obs[i,t,1].item() * (H-1))
        cv2.circle(frame, (x,y), 2, (255,255,255), -1)

    out.write(frame)

out.release()

print("Saved to unified_sim.mp4")