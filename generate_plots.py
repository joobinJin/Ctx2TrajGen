# tools/generate_plots.py
# Usage examples:
#   python generate_plots.py
#   python generate_plots.py --pth-dir pth/pth_full --start 10 --end 200 --step 10
#   python generate_plots.py --only-file pth/pth_noPPO/policy_net_iter90.pth --n-episode 8
#   python generate_plots.py --pth-dir pth/pth_noWGAN-GP
#   python generate_plots.py --pth-dir pth/pth_vanilla
#   python generate_plots.py --pth-dir pth/pth_full --plot-dir plot/custom_out

import os
import sys
import argparse
import glob
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless backend for servers/CI
import matplotlib.pyplot as plt

# --- Import path patch (run from anywhere) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = THIS_DIR if os.path.exists(os.path.join(THIS_DIR, "micro_trajectory.py")) \
       else os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from micro_trajectory import MicroTrajectoryEnv
from models.policy_net import PolicyNetRNN


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def detect_device(pref: str = "auto") -> torch.device:
    """
    Determine execution device.
      - 'cuda': use CUDA if available else CPU
      - 'cpu' : force CPU
      - 'auto': CUDA if available else CPU (default)
    """
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_pth_files(pth_dir: str,
                   start: int,
                   end: int,
                   step: int,
                   include_iter1: bool,
                   only_file: str | None) -> list[str]:
    """
    Build the list of policy checkpoint files to load.
    Priority:
      1) --only-file if present and exists
      2) Range: [start, end] with given step (+ optionally iter1)
      3) Fallback: any policy_net_iter*.pth in pth_dir
    """
    if only_file:
        if os.path.isfile(only_file):
            return [only_file]
        print(f"[WARN] --only-file not found: {only_file}")
        return []

    files = []
    if include_iter1:
        files.append(os.path.join(pth_dir, "policy_net_iter1.pth"))
    for i in range(start, end + 1, step):
        files.append(os.path.join(pth_dir, f"policy_net_iter{i}.pth"))

    files = [f for f in files if os.path.isfile(f)]
    if not files:
        cand = sorted(glob.glob(os.path.join(pth_dir, "policy_net_iter*.pth")))
        if cand:
            print("[INFO] No files matched the requested range; using all found:", len(cand))
            files = cand
    return files


def generate_trajectories(env: MicroTrajectoryEnv,
                          policy: PolicyNetRNN,
                          num_trajs: int,
                          max_steps: int,
                          device: torch.device):
    """
    Roll out trajectories with (policy.sample -> env.step).
    Returns:
      all_states  : list[np.ndarray (T_i+1, 6)]
      all_actions : list[np.ndarray (T_i, 6)]
      seq_lens    : list[int]
    """
    all_states, all_actions, seq_lens = [], [], []

    for epi in range(num_trajs):
        state = env.reset()
        traj_states = [state.copy()]
        traj_actions = []
        hidden = None

        for _ in range(max_steps):
            s_t = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, 6)
            v_ctx, l_ctx, e_ctx = env._get_context()
            ctx_np = np.concatenate([v_ctx, l_ctx, e_ctx], axis=0)  # (39,)
            c_t = torch.tensor(ctx_np, dtype=torch.float32, device=device).view(1, 1, 39)
            seq_len = torch.tensor([1], dtype=torch.long, device=device)

            with torch.no_grad():
                action_t, _, _, _, hidden = policy.sample(s_t, c_t, seq_len, hidden=hidden)

            action = action_t[0, 0].detach().cpu().numpy()
            step_result = env.step(action)
            nxt_s, done = step_result.next_state, step_result.done

            traj_actions.append(action.copy())
            traj_states.append(nxt_s.copy())
            state = nxt_s
            if done:
                break

        all_states.append(np.array(traj_states, dtype=np.float32))
        all_actions.append(np.array(traj_actions, dtype=np.float32))
        seq_lens.append(len(traj_actions))
        print(f"[Trajectory {epi+1}] length={len(traj_actions)}, done={done}")

    return all_states, all_actions, seq_lens


def plot_generated_trajectories(gen_states: list[np.ndarray], n_episodes: int, save_path: str):
    """
    Plot center_x/center_y in pixel scale and save as PNG.
    Assumes center_x, center_y are normalized to [0,1] (3840x2160).
    """
    plt.figure(figsize=(8, 6))
    for i in range(min(n_episodes, len(gen_states))):
        traj = gen_states[i]
        px_x = traj[:, 0] * 3840.0
        px_y = traj[:, 1] * 2160.0
        plt.plot(px_x, px_y, label=f"Ep{i+1}")

    plt.xlabel("center_x (pixels)")
    plt.ylabel("center_y (pixels)")
    plt.title("Generated Trajectories (Pixel Scale)")
    plt.xlim(0, 3840)
    plt.ylim(0, 2160)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate and plot trajectories from trained policies")
    # Data / env
    p.add_argument("--data", type=str, default="Data/clean_DJI.pkl", help="Path to env data (pkl)")
    p.add_argument("--cjson", type=str, default="Data/C.json", help="Path to context JSON")

    # Model files
    p.add_argument("--pth-dir", type=str, default="pth/pth_full",
                   help="Directory containing policy_net_iter*.pth (default: pth/pth_full)")
    p.add_argument("--plot-dir", type=str, default=None,
                   help="Directory to save plots (default: map 'pth/pth_*' → 'plot/plot_*')")
    p.add_argument("--only-file", type=str, default=None,
                   help="Load only this checkpoint file (overrides range options)")
    p.add_argument("--start", type=int, default=10, help="Start iteration (inclusive)")
    p.add_argument("--end", type=int, default=1000, help="End iteration (inclusive)")
    p.add_argument("--step", type=int, default=10, help="Iteration step")
    p.add_argument("--include-iter1", action="store_true", default=True,
                   help="Also include policy_net_iter1.pth (default: True)")
    p.add_argument("--no-iter1", dest="include_iter1", action="store_false",
                   help="Do not include policy_net_iter1.pth")

    # Rollout
    p.add_argument("--n-episode", type=int, default=10, help="Episodes per checkpoint")
    p.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")

    # Device
    p.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto",
                   help="Device: auto/cuda/cpu")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve device
    device = detect_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Resolve plot dir:
    # - If user provides --plot-dir, use it.
    # - Else map pth/pth_full -> plot/plot_full, pth/pth_noPPO -> plot/plot_noPPO, etc.
    if args.plot_dir:
        plot_dir = args.plot_dir
    else:
        base = os.path.basename(args.pth_dir.rstrip("/\\"))
        if base.startswith("pth_"):
            out_base = "plot_" + base[len("pth_"):]  # e.g., 'pth_full' -> 'plot_full'
        else:
            out_base = "plot_" + base                 # fallback: 'foo' -> 'plot_foo'
        plot_dir = os.path.join("plot", out_base)
    os.makedirs(plot_dir, exist_ok=True)

    # Collect checkpoint files
    pth_files = list_pth_files(
        pth_dir=args.pth_dir,
        start=args.start,
        end=args.end,
        step=args.step,
        include_iter1=args.include_iter1,
        only_file=args.only_file,
    )
    if not pth_files:
        print(f"[ERROR] No checkpoint files found in {args.pth_dir}.")
        return
    print(f"[INFO] {len(pth_files)} checkpoint(s) to process.")

    # Build environment once
    env = MicroTrajectoryEnv(args.data, c_json_path=args.cjson, history_length=5)

    # Process each checkpoint
    for ckpt in pth_files:
        print(f"\n=== Processing {ckpt} ===")
        policy = PolicyNetRNN(state_dim=6, context_dim=39, hidden_dim=128, num_layers=2, K=5).to(device)
        state = torch.load(ckpt, map_location=device)
        policy.load_state_dict(state)
        policy.eval()

        gen_states, _, _ = generate_trajectories(
            env, policy, num_trajs=args.n_episode, max_steps=args.max_steps, device=device
        )

        save_name = os.path.splitext(os.path.basename(ckpt))[0] + ".png"
        save_path = os.path.join(plot_dir, save_name)
        plot_generated_trajectories(gen_states, args.n_episode, save_path)
        print(f"[OK] Saved trajectory plot: {save_path}")


if __name__ == "__main__":
    main()
