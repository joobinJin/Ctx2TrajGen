# test/policy_net.py

import os
import sys
import time
import random
import gc
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# --- Ensure project root is on sys.path so imports work when running this file directly ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from micro_trajectory import MicroTrajectoryEnv
from models.policy_net import PolicyNetRNN, ValueNetRNN

# --------------------------------------------------------------
# 0) Config
# --------------------------------------------------------------
BATCH   = 4       # test batch size
MAX_LEN = 300     # max unrolled length for RNN
SEED    = 314
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --------------------------------------------------------------
# 1) Sample a batch from environment
# --------------------------------------------------------------
DATA_PKL = "Data/clean_DJI.pkl"
C_JSON   = "Data/C.json"
env = MicroTrajectoryEnv(DATA_PKL, c_json_path=C_JSON, history_length=5)

states_lst, ctx_lst, len_lst = [], [], []
for i in range(BATCH):
    tr = env.trajs[i]
    T  = min(tr["seq_len"], MAX_LEN)

    # (T, 6)
    s_np = tr["states"][:T]
    # (T, 39) = vehicle(30) + lane(3) + exists(6)
    c_np = np.hstack([
        tr["vehicle_context"][:T],
        tr["lane_context"][:T],
        tr["exists_context"][:T],
    ])

    states_lst.append(torch.tensor(s_np, dtype=torch.float32))
    ctx_lst.append(torch.tensor(c_np, dtype=torch.float32))
    len_lst.append(T)

# pad_sequence -> (B, T_max, 6)/(B, T_max, 39)
states_pad = pad_sequence(states_lst, batch_first=True)
ctx_pad    = pad_sequence(ctx_lst, batch_first=True)
seq_lens   = torch.tensor(len_lst, dtype=torch.long)

states_pad, ctx_pad, seq_lens = [x.to(DEVICE) for x in (states_pad, ctx_pad, seq_lens)]

print(f"[Batch] states={states_pad.shape}, ctx={ctx_pad.shape}, len={len_lst}")

# --------------------------------------------------------------
# 2) Build models
# --------------------------------------------------------------
policy = PolicyNetRNN(
    state_dim=6,
    context_dim=39,
    hidden_dim=128,
    num_layers=2,
    K=5
).to(DEVICE).eval()

# ValueNetRNN returns (B, T, 1)
value = ValueNetRNN(
    state_dim=6,
    context_dim=39,
    hidden_dim=128,
    num_layers=2
).to(DEVICE).eval()

if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats(DEVICE)
t0 = time.time()

# --------------------------------------------------------------
# 2-A) Forward & sample & value
# --------------------------------------------------------------
with torch.no_grad():
    # (a) policy.forward
    mu, log_sig, pi, _ = policy.forward(states_pad, ctx_pad, seq_lens)
    print("\n=== Policy forward ===")
    print("mu       :", mu.shape, "     (B, T, K, 6)")
    print("log_sig  :", log_sig.shape, "(B, T, K, 6)")
    print("pi       :", pi.shape,      "(B, T, K)")
    print("mu range      =", (mu.min().item(), mu.max().item()))
    print("log_sig range =", (log_sig.min().item(), log_sig.max().item()))
    print("pi sum per step (should be ~1):")
    print(pi.sum(dim=-1))

    # NaN/Inf guards
    if torch.isnan(mu).any() or torch.isinf(mu).any():
        raise ValueError("NaN or Inf detected in mu")
    if torch.isnan(log_sig).any() or torch.isinf(log_sig).any():
        raise ValueError("NaN or Inf detected in log_sig")
    if torch.isnan(pi).any() or torch.isinf(pi).any():
        raise ValueError("NaN or Inf detected in pi")

    # (b) policy.sample
    act_sample, mu_sel, log_s_sel, comp_sel, _ = policy.sample(states_pad, ctx_pad, seq_lens)
    print("\n=== Policy sample ===")
    print("sample   :", act_sample.shape, "  range=",
          (act_sample.min().item(), act_sample.max().item()))
    print("mu_sel   :", mu_sel.shape)
    print("log_s_sel:", log_s_sel.shape)
    print("comp_sel :", comp_sel.shape, comp_sel[:3])

    if torch.isnan(act_sample).any() or torch.isinf(act_sample).any():
        raise ValueError("NaN or Inf detected in act_sample")

    # (c) value forward -> (B, T, 1)
    v_out, _ = value(states_pad, ctx_pad, seq_lens)
    print("\n=== Value ===")
    print("v_out shape :", v_out.shape, "  (B, T, 1)")

    # Per-batch valid (unpadded) value slice
    for i_b in range(v_out.size(0)):
        real_T = seq_lens[i_b].item()
        val_part = v_out[i_b, :real_T, 0].cpu().numpy()
        print(f" batch{i_b}: Value first ~3 steps = {val_part[:3]} ... (len={real_T})")

dt = (time.time() - t0) * 1e3
mem_peak = (torch.cuda.max_memory_allocated(DEVICE) / 1e6) if DEVICE.type == "cuda" else 0
print(f"\nForward time {dt:4.1f} ms | CUDA peak={mem_peak:.1f} MB")

# --------------------------------------------------------------
# 2-B) Shape assertions
# --------------------------------------------------------------
B = states_pad.size(0)
T_max = states_pad.size(1)
assert mu.size()       == (B, T_max, 5, 6), "mu shape mismatch"
assert log_sig.size()  == (B, T_max, 5, 6), "log_sig shape mismatch"
assert pi.size()       == (B, T_max, 5),    "pi shape mismatch"
assert act_sample.size()== (B, T_max, 6),   "action shape mismatch"
assert mu_sel.size()   == (B, T_max, 6),    "mu_sel shape mismatch"
assert log_s_sel.size()== (B, T_max, 6),    "log_s_sel shape mismatch"
assert comp_sel.size() == (B, T_max),       "comp_sel shape mismatch"
assert v_out.size()    == (B, T_max, 1),    "value shape mismatch"

# --------------------------------------------------------------
# 3) 150-step rollout test (log action/state stats)
# --------------------------------------------------------------
def run_rollout(env, policy, max_steps=150):
    env.reset()
    belief = env.get_belief_state()  # shape (history_len, 45)
    if belief.shape[0] < 1:
        print("\n[No belief state length??]")
        return

    # initial state & context from the last row of belief
    s_last = torch.tensor(belief[-1, :6], dtype=torch.float32).view(1, 1, 6).to(DEVICE)
    c_last = torch.tensor(belief[-1, 6:], dtype=torch.float32).view(1, 1, 39).to(DEVICE)
    l_last = torch.tensor([1], device=DEVICE)
    hidden = None

    rollout_states = []
    rollout_rewards = []
    rollout_actions = []
    done = False

    for step in range(max_steps):
        with torch.no_grad():
            a, _, _, _, hidden = policy.sample(s_last, c_last, l_last, hidden=hidden)
        a = a.squeeze().cpu().numpy()  # (6,)
        rollout_actions.append(a.copy())

        step_result = env.step(a)
        last_s = step_result.cur_state
        nxt_s = step_result.next_state
        r = step_result.reward
        d = step_result.done
        m = step_result.mask
        next_context = step_result.next_context

        # Guards
        if len(nxt_s) != 6:
            raise ValueError(f"Step {step}: Expected next_state length 6, got {len(nxt_s)}")
        if len(next_context) != 39:
            raise ValueError(f"Step {step}: Expected next_context length 39, got {len(next_context)}")
        if np.any(np.isnan(nxt_s)) or np.any(np.isinf(nxt_s)):
            raise ValueError(f"Step {step}: NaN/Inf in next_state")
        if np.any(np.isnan(r)) or np.any(np.isinf(r)):
            raise ValueError(f"Step {step}: NaN/Inf in reward")

        # Range checks (center_x/y in [0,1]; v_x/a_x clipped to dataset-derived bounds)
        if not (0 <= nxt_s[0] <= 1 and 0 <= nxt_s[1] <= 1):
            print(f"Warning: Step {step} - center_x or center_y out of range: {nxt_s[0]:.3f}, {nxt_s[1]:.3f}")
        if not (env.v_x_lower <= nxt_s[2] <= env.v_x_upper):
            print(f"Warning: Step {step} - v_x out of range: {nxt_s[2]:.3f}")
        if not (env.a_x_lower <= nxt_s[4] <= env.a_x_upper):
            print(f"Warning: Step {step} - a_x out of range: {nxt_s[4]:.3f}")

        rollout_states.append(nxt_s.copy())
        rollout_rewards.append(r)
        print(f"Step {step}: reward={r:.3f}, done={d}, mask={m}, "
              f"last_s=[{last_s[0]:.6f}, {last_s[1]:.6f}] -> "
              f"next_s=[{nxt_s[0]:.6f}, {nxt_s[1]:.6f}]")

        if d:
            print(f"\nEpisode ended at step {step} with reward sum: {sum(rollout_rewards):.3f}")
            done = True
            break

        # to next step
        s_last = torch.tensor(nxt_s, dtype=torch.float32).view(1, 1, 6).to(DEVICE)
        c_last = torch.tensor(next_context, dtype=torch.float32).view(1, 1, 39).to(DEVICE)
        l_last = torch.tensor([1], device=DEVICE)

    if not done:
        print(f"\nRollout completed {max_steps} steps with reward sum: {sum(rollout_rewards):.3f}")

    # Trajectory stats
    if rollout_states:
        rollout_states = np.array(rollout_states)
        print(f"\nRollout trajectory stats:")
        print(f"  center_x: min={rollout_states[:, 0].min():.3f}, max={rollout_states[:, 0].max():.3f}, "
              f"mean={rollout_states[:, 0].mean():.3f}, std={rollout_states[:, 0].std():.3f}")
        print(f"  center_y: min={rollout_states[:, 1].min():.3f}, max={rollout_states[:, 1].max():.3f}, "
              f"mean={rollout_states[:, 1].mean():.3f}, std={rollout_states[:, 1].std():.3f}")
        print(f"  v_x: min={rollout_states[:, 2].min():.3f}, max={rollout_states[:, 2].max():.3f}, "
              f"mean={rollout_states[:, 2].mean():.3f}, std={rollout_states[:, 2].std():.3f}")
        print(f"  a_x: min={rollout_states[:, 4].min():.3f}, max={rollout_states[:, 4].max():.3f}, "
              f"mean={rollout_states[:, 4].mean():.3f}, std={rollout_states[:, 4].std():.3f}")
        print(f"Expert data std for comparison: center_x=0.0051, v_x=0.2586, a_x=1.1493")

    # Action stats
    if rollout_actions:
        rollout_actions = np.array(rollout_actions)  # (num_steps, 6)
        action_std = rollout_actions.std(axis=0)
        print(f"\nAction stats:")
        print(f"  center_x std={action_std[0]:.6f}, center_y std={action_std[1]:.6f}")
        print(f"  v_x std={action_std[2]:.6f}, v_y std={action_std[3]:.6f}")
        print(f"  a_x std={action_std[4]:.6f}, a_y std={action_std[5]:.6f}")
        print(f"Expert data std for comparison: center_x=0.0051, v_x=0.2586, a_x=1.1493")

print("\n=== 150-Step Rollout ===")
run_rollout(env, policy)

# --------------------------------------------------------------
# 4) action_scale tests (vary scale and observe rewards)
# --------------------------------------------------------------
action_scales = [0.01, 0.03, 0.05, 0.1, 0.2]
print("\n=== Action Scale Tests ===")
env.reset()
belief = env.get_belief_state()
if belief.shape[0] >= 1:
    s_last = torch.tensor(belief[-1, :6], dtype=torch.float32).view(1, 1, 6).to(DEVICE)
    c_last = torch.tensor(belief[-1, 6:], dtype=torch.float32).view(1, 1, 39).to(DEVICE)
    l_last = torch.tensor([1], device=DEVICE)
    hidden = None

    for scale in action_scales:
        with torch.no_grad():
            scale_tensor = policy.action_scale.to(DEVICE) * scale  # per-dim scale preserved
            a, _, _, _, hidden = policy.sample(s_last, c_last, l_last, action_scale=scale_tensor, hidden=hidden)
        a = a.squeeze().cpu().numpy()
        step_result = env.step(a)
        last_s = step_result.cur_state
        nxt_s = step_result.next_state
        r = step_result.reward
        d = step_result.done
        m = step_result.mask
        next_context = step_result.next_context

        if np.any(np.isnan(nxt_s)) or np.any(np.isinf(nxt_s)):
            raise ValueError(f"Action Scale={scale}: NaN/Inf in next_state")
        if np.any(np.isnan(r)) or np.any(np.isinf(r)):
            raise ValueError(f"Action Scale={scale}: NaN/Inf in reward")

        print(f"Action Scale={scale}: reward={r:.3f}, done={d}, mask={m}, "
              f"last_s=[{last_s[0]:.6f}, {last_s[1]:.6f}] -> "
              f"next_s=[{nxt_s[0]:.6f}, {nxt_s[1]:.6f}]")

        # update for next step
        s_last = torch.tensor(nxt_s, dtype=torch.float32).view(1, 1, 6).to(DEVICE)
        c_last = torch.tensor(next_context, dtype=torch.float32).view(1, 1, 39).to(DEVICE)
        l_last = torch.tensor([1], device=DEVICE)

# --------------------------------------------------------------
# 5) Random forward passes with varied seq_lens
# --------------------------------------------------------------
print("\n=== Random forward passes ===")
for _ in range(3):
    seq_debug = seq_lens.clone()
    for i in range(seq_debug.size(0)):
        seq_debug[i] = np.random.randint(1, seq_lens[i].item() + 1)

    with torch.no_grad():
        mu_, log_sig_, pi_, _ = policy.forward(states_pad, ctx_pad, seq_debug)
        v_, _ = value(states_pad, ctx_pad, seq_debug)
    print(f" seq_debug={seq_debug.tolist()}, mu_.shape={mu_.shape}, v_.shape={v_.shape}")

    if torch.isnan(mu_).any() or torch.isinf(mu_).any():
        raise ValueError("NaN or Inf detected in policy forward output")
    if torch.isnan(v_).any() or torch.isinf(v_).any():
        raise ValueError("NaN or Inf detected in value forward output")

# --------------------------------------------------------------
# 6) Clean-up
# --------------------------------------------------------------
del policy, value, states_pad, ctx_pad
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n[ OK ] policy_net & value_net RNN test completed.\n")
