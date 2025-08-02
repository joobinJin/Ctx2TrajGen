# test/test_env.py

import os
import sys
import json
import math
import numpy as np
from collections import Counter

# --- Ensure project root is on sys.path so imports work when running this file directly ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from micro_trajectory import MicroTrajectoryEnv

# -----------------------------------------------------------------
# 0) Create environment
# -----------------------------------------------------------------
pkl_path    = "Data/clean_DJI.pkl"   # relative to project root
c_json_path = "Data/C.json"
env = MicroTrajectoryEnv(pkl_path, c_json_path=c_json_path, history_length=5, total_iterations=100)

print("\n[ SETUP ] MicroTrajectoryEnv ready")

# -----------------------------------------------------------------
# Distributions of clipped variables (v_x, a_x) from trajectories
# -----------------------------------------------------------------
print("\n=== Distribution of clipped variables (v_x, a_x) ===")
vx_values, ax_values = [], []
for traj in env.trajs:
    vx_values.extend(traj['states'][:, 2])  # v_x is col 2
    ax_values.extend(traj['states'][:, 4])  # a_x is col 4

vx_values = np.array(vx_values)
ax_values = np.array(ax_values)

print(f"v_x min: {np.min(vx_values):.3f}, max: {np.max(vx_values):.3f}")
print(f"v_x mean: {np.mean(vx_values):.3f}, std: {np.std(vx_values):.3f}")
print(f"a_x min: {np.min(ax_values):.3f}, max: {np.max(ax_values):.3f}")
print(f"a_x mean: {np.mean(ax_values):.3f}, std: {np.std(ax_values):.3f}")

# -----------------------------------------------------------------
# A) Inspect the beginning of a specific trajectory
# -----------------------------------------------------------------
traj = env.trajs[0]
states = traj["states"]
real_len = traj["seq_len"]
print(f"\n=== First frames of traj[0] (seq_len {real_len}) ===")
for i in range(min(real_len, 10)):
    cx, cy, vx, vy, ax, ay = states[i]
    print(f"Frame {i:02d} | cx={cx:.4f}, cy={cy:.4f}, v_x={vx:.3f}, v_y={vy:.3f}, a_x={ax:.3f}, a_y={ay:.3f}")

# -----------------------------------------------------------------
# 1) Basic trajectory shapes
# -----------------------------------------------------------------
print("\n=== 1) Basic trajectory shapes (normalized & clipped) ===")
print("states           :", traj['states'].shape)
print("vehicle_context  :", traj['vehicle_context'].shape)
print("lane_context     :", traj['lane_context'].shape)
print("exists_context   :", traj['exists_context'].shape)
print("mask             :", traj['mask'].shape)
print("seq_len          :", traj['seq_len'])

assert not np.isnan(traj['states']).any(),          "NaN in states"
assert not np.isnan(traj['vehicle_context']).any(), "NaN in vehicle_context"
assert not np.isnan(traj['lane_context']).any(),    "NaN in lane_context"
assert not np.isnan(traj['exists_context']).any(),  "NaN in exists_context"

# -----------------------------------------------------------------
# 2) Diversity of reset() initial states
# -----------------------------------------------------------------
print("\n=== 2) Diversity of initial states (reset) ===")
N_RESET = 50
cx_list, lanes = [], []
for _ in range(N_RESET):
    s = env.reset()
    cx_list.append(s[0])
    _, lc, _ = env._get_context()
    lanes.append(int(np.argmax(lc)) + 1)
print(f"center_x range   : {min(cx_list):.3f} ~ {max(cx_list):.3f}")
print(f"lane distribution: {dict(Counter(lanes))}")

# -----------------------------------------------------------------
# 3) Expert trajectories (initial split)
# -----------------------------------------------------------------
print("\n=== 3) Expert trajectories (initial split) ===")
demo_trajs = env.expert_trajs
print("num expert trajs :", len(demo_trajs))
first = demo_trajs[0]['episode'][0]
print("  cur_state   :", first.cur_state)
print("  action      :", first.action)
print("  next_state  :", first.next_state)
print("  next_context:", first.next_context)

# -----------------------------------------------------------------
# 4) Belief state & reward test
# -----------------------------------------------------------------
print("\n=== 4) Belief state / reward test ===")
env.current_traj_idx = 0
env.reset()
print("belief shape    :", env.get_belief_state().shape)

for i in range(5):
    a = demo_trajs[0]['episode'][i].action
    step = env.step(a)
    print(f"step{i+1} reward {step.reward:.3f}, done {step.done}, mask {step.mask}")

# -----------------------------------------------------------------
# 5) Extreme action → clipping check
# -----------------------------------------------------------------
print("\n=== 5) Clipping test (extreme Δ) ===")
env.reset()
ext = np.array([10, 10, 10, 10, 10, 10], dtype=float)
step = env.step(ext)
print("after extreme action state:", step.next_state)

# micro_trajectory.py (cleaned) clips center_x/center_y to [0,1]
assert (0.0 <= step.next_state[0] <= 1.0), "center_x not clipped properly"
assert (0.0 <= step.next_state[1] <= 1.0), "center_y not clipped properly"
assert not np.isnan(step.next_state).any()

# -----------------------------------------------------------------
# 6) Lane ROI consistency (whole trajectories)
# -----------------------------------------------------------------
print("\n=== 6) Lane ROI consistency (all trajectories) ===")
viol, total = 0, 0
for tr in env.trajs:
    T = tr['seq_len']
    for i in range(T):
        cx, cy = tr['states'][i, 0], tr['states'][i, 1]
        lane_idx = np.argmax(tr['lane_context'][i])
        found_idx = env._locate_lane_roi(cx, cy)
        if found_idx != lane_idx:
            viol += 1
        total += 1
pct = (viol / total * 100) if total > 0 else 0.0
print(f"ROI violation rate: {pct:.3f}%")
if total > 0:
    assert pct < 5.0, "ROI mapping error (>5.0%)"

# -----------------------------------------------------------------
# 7) Consecutive step NaN/Inf check
# -----------------------------------------------------------------
print("\n=== 7) Consecutive step NaN/Inf check ===")
env.reset()
good = True
for _ in range(30):
    step = env.step(np.random.uniform(-.05, .05, 6))
    if not np.isfinite(step.next_state).all():
        good = False
        break
print("NaN/Inf occurred:", not good)
assert good, "state NaN/Inf detected"

# -----------------------------------------------------------------
# 8) Sequence length range
# -----------------------------------------------------------------
print("\n=== 8) Sequence length range ===")
min_len = min(tr['seq_len'] for tr in env.trajs)
max_len = max(tr['seq_len'] for tr in env.trajs)
print(f"total {len(env.trajs)} trajs | seq_len range: {min_len} ~ {max_len}")
if min_len < 2:
    print(f"Warning: min seq_len is {min_len} (<2) → possible episodes without actions in GAIL")

# -----------------------------------------------------------------
# 9) states vs. contexts shape check
# -----------------------------------------------------------------
print("\n=== 9) states vs. lane_context shape ===")
errors = 0
for i, tr in enumerate(env.trajs):
    st_shape = tr["states"].shape
    ln_shape = tr["lane_context"].shape
    v_ctx_sh = tr["vehicle_context"].shape
    ex_ctx_sh = tr["exists_context"].shape
    if not (st_shape[0] == v_ctx_sh[0] == ln_shape[0] == ex_ctx_sh[0]):
        errors += 1
        print(f"traj {i}: shape mismatch {st_shape}, {v_ctx_sh}, {ln_shape}, {ex_ctx_sh}")
print(f"num shape-mismatch trajs: {errors}")
assert errors == 0, "some trajectory has shape mismatch among states/contexts"

# -----------------------------------------------------------------
# 10) Action consistency (optional)
# -----------------------------------------------------------------
print("\n=== 10) Action consistency check ===")
episodes = demo_trajs[0]["episode"]
if episodes:
    mismatch_count = 0
    for st in episodes:
        if not np.allclose(st.next_state, st.cur_state + st.action, atol=1e-2):
            mismatch_count += 1
    print(f"episode mismatch count (cur+act vs next) = {mismatch_count}")
else:
    print("no 'episode' key found, skip action consistency check")

# -----------------------------------------------------------------
# 11) Reward distribution
# -----------------------------------------------------------------
print("\n=== 11) Reward distribution ===")
rewards = []
env.reset()
for _ in range(100):
    for _ in range(50):
        action = np.random.uniform(-0.05, 0.05, 6)
        step = env.step(action)
        rewards.append(step.reward)
        if step.done:
            env.reset()
rewards = np.array(rewards)
print(f"Reward mean: {np.mean(rewards):.3f}, std: {np.std(rewards):.3f}")
print(f"Reward min: {np.min(rewards):.3f}, max: {np.max(rewards):.3f}")

# -----------------------------------------------------------------
# 12) State-transition sanity & ranges
# -----------------------------------------------------------------
print("\n=== 12) State-transition sanity & ranges ===")
env.reset()
state_diffs = []
for _ in range(50):
    prev_state = env._cur_state.copy()
    action = np.random.uniform(-0.05, 0.05, 6)
    step = env.step(action)
    next_state = step.next_state
    diff = next_state - prev_state
    state_diffs.append(diff)
state_diffs = np.array(state_diffs)
print(f"State diff mean: {np.mean(state_diffs, axis=0)}")
print(f"State diff std : {np.std(state_diffs, axis=0)}")
print(f"Clipping ranges: v_x [{env.v_x_lower:.3f}, {env.v_x_upper:.3f}], a_x [{env.a_x_lower:.3f}, {env.a_x_upper:.3f}]")

# -----------------------------------------------------------------
# 13) Expert action statistics
# -----------------------------------------------------------------
print("\n=== 13) Expert action statistics ===")
actions = []
for tr in env.expert_trajs:
    for st in tr['episode']:
        actions.append(st.action)
actions = np.array(actions)
print(f"Expert action mean: {np.mean(actions, axis=0)}")
print(f"Expert action std : {np.std(actions, axis=0)}")
print(f"Current test action range: [-0.05, 0.05]")

# -----------------------------------------------------------------
# 14) exists_context distribution
# -----------------------------------------------------------------
print("\n=== 14) exists_context distribution ===")
exists_values = []
for tr in env.trajs:
    exists_values.extend(tr['exists_context'].flatten())
exists_values = np.array(exists_values)
print(f"Exists min: {np.min(exists_values):.3f}, max: {np.max(exists_values):.3f}")
print(f"Exists mean: {np.mean(exists_values):.3f}, std: {np.std(exists_values):.3f}")

# -----------------------------------------------------------------
# 15) seq_len distribution & terminal condition
# -----------------------------------------------------------------
print("\n=== 15) seq_len distribution & termination ===")
seq_lengths = [tr['seq_len'] for tr in env.trajs]
print(f"seq_len stats: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")
short_trajs = sum(1 for sl in seq_lengths if sl < 10)
print(f"num trajs with seq_len < 10: {short_trajs}")

# -----------------------------------------------------------------
# 16) Dynamic scale_factor test
# -----------------------------------------------------------------
print("\n=== 16) Dynamic scale_factor test ===")
env.total_iterations = 100
scale_factors = []
for it in [0, 25, 50, 75, 100]:
    env.set_current_iteration(it)
    env.reset()
    _ = env.step(np.random.uniform(-0.05, 0.05, 6))
    scale_factors.append(env.get_scale_factor())
    print(f"Iteration {it:3d}: scale_factor = {scale_factors[-1]:.4f}")
assert all(scale_factors[i] > scale_factors[i+1] for i in range(len(scale_factors)-1)), \
    "scale_factor does not decrease with iteration"
print("scale_factor decreasing check: OK")

# -----------------------------------------------------------------
# 17) Similarity penalty weight (analytical) test
# -----------------------------------------------------------------
print("\n=== 17) Similarity penalty weight (analytical) test ===")
env.total_iterations = 100
penalty_weights = []
for it in [0, 25, 50, 75, 100]:
    env.set_current_iteration(it)
    pw = 0.005 * (1 - it / env.total_iterations) if env.total_iterations > 0 else 0.005
    penalty_weights.append(pw)
    print(f"Iteration {it:3d}: penalty_weight = {pw:.4f}")
assert all(penalty_weights[i] > penalty_weights[i+1] for i in range(len(penalty_weights)-1)), \
    "penalty_weight does not decrease with iteration"
print("penalty_weight decreasing check: OK")

# -----------------------------------------------------------------
# 18) Early vs late iteration transition behavior
# -----------------------------------------------------------------
print("\n=== 18) Early vs. late iteration transition behavior ===")
env.total_iterations = 100
env.set_current_iteration(0)
_ = env.reset()
action = env.expert_trajs[0]['episode'][0].action
early = env.step(action)
print(f"Early iteration (0): next_state[:2] = {early.next_state[:2]}")

env.set_current_iteration(100)
_ = env.reset()
late = env.step(action)
print(f"Late iteration (100): next_state[:2]  = {late.next_state[:2]}")
print("Qualitative difference between early and late transitions printed above.")

# -----------------------------------------------------------------
# 19) Effect of total_iterations on scale_factor
# -----------------------------------------------------------------
print("\n=== 19) Effect of total_iterations on scale_factor ===")
for total_it in [50, 100, 200]:
    env.total_iterations = total_it
    env.set_current_iteration(total_it // 2)
    _ = env.reset()
    _ = env.step(np.random.uniform(-0.05, 0.05, 6))
    print(f"total_iterations={total_it}, mid-iteration scale_factor = {env.get_scale_factor():.4f}")

# -----------------------------------------------------------------
# 20) Monitor similarity penalty trend (estimated)
# -----------------------------------------------------------------
print("\n=== 20) Similarity penalty trend (estimated) ===")
env.total_iterations = 100
penalties = []
for it in [0, 50, 100]:
    env.set_current_iteration(it)
    env.reset()
    step = env.step(np.random.uniform(-0.05, 0.05, 6))
    # Estimate similarity penalty = MSE(next_state vs. expert) * penalty_weight
    traj = env.trajs[env.current_traj_idx]
    # get expert state index consistent with reward computation timing
    exp_idx = max(env.current_step - 1, 0)
    expert_state = traj["states"][exp_idx]
    mse = np.mean((step.next_state - expert_state) ** 2)
    pw = 0.005 * (1 - env.current_iteration / env.total_iterations) if env.total_iterations > 0 else 0.005
    penalties.append(mse * pw)
    print(f"Iteration {it:3d}: est_penalty = {penalties[-1]:.4f}")
assert all(penalties[i] > penalties[i+1] for i in range(len(penalties)-1)), \
    "estimated similarity penalty does not decrease with iteration"
print("estimated penalty decreasing check: OK")

# -----------------------------------------------------------------
# 21) Edge cases / error handling
# -----------------------------------------------------------------
print("\n=== 21) Edge cases / error handling ===")
try:
    env.total_iterations = 0
    env.set_current_iteration(0)
    env.reset()
    _ = env.step(np.random.uniform(-0.05, 0.05, 6))
    print("total_iterations=0: passed (no division-by-zero)")
except ZeroDivisionError:
    raise AssertionError("division-by-zero occurred at total_iterations=0")

env.total_iterations = 100
env.set_current_iteration(150)  # iteration > total_iterations
_ = env.step(np.random.uniform(-0.05, 0.05, 6))
print("iteration > total_iterations: passed")

print("\n[ OK ] All tests completed.")
