# test/test_discriminator.py

import os
import sys
import random
import gc
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# --- Add project root to sys.path so imports work when running this file directly ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from micro_trajectory import MicroTrajectoryEnv
from models.discriminator import DiscriminatorRNN
from models.policy_net import PolicyNetRNN

# -----------------------------------------------------------------------------
# 0) Config
# -----------------------------------------------------------------------------
BATCH = 4
MAX_LEN = 300
SEED = 314
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------------------------------------------------------------
# 1) Sample a mini-batch from environment trajectories (Expert → real)
# -----------------------------------------------------------------------------
DATA_PKL = "Data/clean_DJI.pkl"
C_JSON = "Data/C.json"
env = MicroTrajectoryEnv(DATA_PKL, c_json_path=C_JSON, history_length=5)

s_list, c_list, a_list, len_list = [], [], [], []
for i in range(BATCH):
    traj = env.trajs[i]
    T = min(traj["seq_len"], MAX_LEN)

    s_np = traj["states"][:T]  # (T, 6)
    ctx_np = np.hstack([
        traj["vehicle_context"][:T],
        traj["lane_context"][:T],
        traj["exists_context"][:T]
    ])  # (T, 39)

    # Actions = states[t+1] - states[t]
    T_act = max(T - 1, 0)
    a_np = (s_np[1:] - s_np[:-1]) if T_act > 0 else np.zeros((0, 6), dtype=np.float32)
    # Add action clipping for safety
    a_np = np.clip(a_np, -1.5, 1.5)

    # NaN check
    if np.any(np.isnan(s_np)) or np.any(np.isnan(ctx_np)) or np.any(np.isnan(a_np)):
        print(f"Warning: NaN detected in batch {i}")

    s_list.append(torch.tensor(s_np, dtype=torch.float32))
    c_list.append(torch.tensor(ctx_np, dtype=torch.float32))
    a_list.append(torch.tensor(a_np, dtype=torch.float32))
    len_list.append(T_act)

def pad_tensor_list(lst):
    """Pad a list of (T, ...) tensors to (B, T_max, ...) and move to DEVICE."""
    return pad_sequence(lst, batch_first=True).to(DEVICE)

S_pad  = pad_tensor_list(s_list)   # (B, T, 6)
C_pad  = pad_tensor_list(c_list)   # (B, T, 39)
A_pad  = pad_tensor_list(a_list)   # (B, T-1, 6)
L_tens = torch.tensor(len_list, dtype=torch.long, device=DEVICE)

print(f"\n[Batch] states={S_pad.shape}, ctx={C_pad.shape}, act={A_pad.shape}, len={L_tens.tolist()}")

# -----------------------------------------------------------------------------
# 2) Build DiscriminatorRNN
# -----------------------------------------------------------------------------
disc = DiscriminatorRNN(
    state_dim=6,
    context_dim=39,
    action_dim=6,
    hidden_dim=128,
    num_layers=2
).to(DEVICE)

# -----------------------------------------------------------------------------
# 3) Forward (real) in no_grad
# -----------------------------------------------------------------------------
with torch.no_grad():
    d_out = disc(S_pad, C_pad, A_pad, L_tens)
    print(f"\n=== forward ===\nD(s,c,a) shape={tuple(d_out.shape)}, example={d_out[0].item():.4f}")

    # NaN/Inf check
    if torch.isnan(d_out).any() or torch.isinf(d_out).any():
        raise ValueError("NaN or Inf detected in D(s,c,a) output")

    rwd = disc.reward(S_pad, C_pad, A_pad, L_tens)
    print(f"reward[0] = {rwd[0].item():.4f}")

# -----------------------------------------------------------------------------
# 4) Gradient penalty sanity check
# -----------------------------------------------------------------------------
disc.train()
S_fake = (S_pad + 0.01 * torch.randn_like(S_pad)).clamp(-1, 1).detach().requires_grad_(True)
C_fake = (C_pad + 0.01 * torch.randn_like(C_pad)).detach().requires_grad_(True)
A_fake = (A_pad + 0.01 * torch.randn_like(A_pad)).clamp(-1, 1).detach().requires_grad_(True)

gp_list = []
for _ in range(10):
    gp = disc.gradient_penalty(
        real_tup=(S_pad, C_pad, A_pad),
        fake_tup=(S_fake, C_fake, A_fake),
        seq_lens=L_tens,
        device=DEVICE
    )
    gp_list.append(gp.item())

    if torch.isnan(gp).any() or torch.isinf(gp).any():
        raise ValueError("NaN or Inf detected in gradient penalty")

mean_gp, std_gp = np.mean(gp_list), np.std(gp_list)
print("\n=== gradient-penalty check ===")
print(f"||∇||² penalty: mean={mean_gp:.4f}, std={std_gp:.4f}")
if std_gp > 0.1:
    print("Warning: High variance in gradient penalty!")

# -----------------------------------------------------------------------------
# 5) Compare real vs. fake using PolicyNetRNN-generated actions
# -----------------------------------------------------------------------------
policy = PolicyNetRNN(
    state_dim=6,
    context_dim=39,
    hidden_dim=128,
    num_layers=2,
    K=5
).to(DEVICE)

with torch.no_grad():
    act_fake, _, _, _, _ = policy.sample(S_pad, C_pad, L_tens)
    d_real = disc(S_pad, C_pad, A_pad, L_tens)
    d_fake = disc(S_pad, C_pad, act_fake, L_tens)
    print("\n=== Real vs. Fake Discrimination ===")
    print(f"D(real) mean={d_real.mean().item():.4f}, std={d_real.std().item():.4f}")
    print(f"D(fake) mean={d_fake.mean().item():.4f}, std={d_fake.std().item():.4f}")
    diff = (d_real - d_fake).abs().mean().item()
    print(f"D(real) - D(fake) |mean diff| = {diff:.4f}")
    if diff < 0.01:
        print("Warning: Discriminator struggles to distinguish real vs fake!")

# -----------------------------------------------------------------------------
# 6) Random seq_lens test
# -----------------------------------------------------------------------------
print("\n=== random seq_lens test ===")
for _ in range(2):
    seq_rand = L_tens.clone()
    for i in range(seq_rand.size(0)):
        seq_rand[i] = np.random.randint(1, seq_rand[i].item() + 1)
    d_out_rand = disc(S_pad, C_pad, A_pad, seq_rand)
    print(f" seq_rand={seq_rand.tolist()}, D_out shape={tuple(d_out_rand.shape)}")

# -----------------------------------------------------------------------------
# 7) Extreme seq_lens test
# -----------------------------------------------------------------------------
print("\n=== extreme seq_lens test ===")
extreme_cases = [
    torch.ones_like(L_tens),                # min (1)
    L_tens.clone() * 0 + A_pad.size(1),     # max (T-1)
]
for seq_rand in extreme_cases:
    try:
        d_out_rand = disc(S_pad, C_pad, A_pad, seq_rand)
        print(f" seq_rand={seq_rand.tolist()}, D_out shape={tuple(d_out_rand.shape)}")
    except Exception as e:
        print(f" Error with seq_rand={seq_rand.tolist()}: {e}")

# -----------------------------------------------------------------------------
# 8) Data consistency checks
# -----------------------------------------------------------------------------
print("\n=== data consistency check ===")
assert S_pad.shape[1] == C_pad.shape[1], "State and context length mismatch!"
assert A_pad.shape[1] == S_pad.shape[1] - 1, "Action length mismatch!"
for i in range(BATCH):
    if L_tens[i] > A_pad.shape[1]:
        raise ValueError(f"seq_lens[{i}]={L_tens[i]} exceeds action length {A_pad.shape[1]}!")

# -----------------------------------------------------------------------------
# 9) Device consistency
# -----------------------------------------------------------------------------
print("\n=== device consistency check ===")
for name, tensor in [("S_pad", S_pad), ("C_pad", C_pad), ("A_pad", A_pad), ("L_tens", L_tens)]:
    print(f"{name} device: {tensor.device}")

# -----------------------------------------------------------------------------
# 10) Memory usage
# -----------------------------------------------------------------------------
print("\n=== memory usage ===")
if torch.cuda.is_available():
    print(f"Current GPU memory: {torch.cuda.memory_allocated(DEVICE)/1e6:.2f} MB")

# -----------------------------------------------------------------------------
# 11) Input range sanity
# -----------------------------------------------------------------------------
print("\n=== input range check ===")
for name, tensor in [("S_pad", S_pad), ("A_pad", A_pad)]:
    min_val, max_val = tensor.min().item(), tensor.max().item()
    print(f"{name}: min={min_val:.4f}, max={max_val:.4f}")
    if min_val < -1.5 or max_val > 1.5:
        print(f"Warning: {name} exceeds expected range!")

# -----------------------------------------------------------------------------
# 12) Clean-up
# -----------------------------------------------------------------------------
del disc, policy, S_pad, C_pad, A_pad
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n[ OK ] Discriminator RNN test completed.\n")
