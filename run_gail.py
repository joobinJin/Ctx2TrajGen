import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn

# Keep path hack for current repo layout (replace with package imports when refactoring)
sys.path.append(os.path.dirname(os.getcwd()))

from micro_trajectory import MicroTrajectoryEnv
from models.policy_net import PolicyNetRNN, ValueNetRNN
from models.discriminator import DiscriminatorRNN
from utils.utils import get_gae

cudnn.enabled = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


def argparser():
    parser = argparse.ArgumentParser(description="GAIL Training Script")
    parser.add_argument('--iteration', type=int, default=200, help="Number of training iterations")
    parser.add_argument('--n_episode', type=int, default=10, help="Number of episodes per iteration")
    parser.add_argument('--max_steps', type=int, default=300, help="Maximum steps per episode")
    parser.add_argument('-b', '--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('-nh', '--hidden', type=int, default=128, help="Hidden dimension")
    parser.add_argument('--num-layers', type=int, default=2, help="Number of RNN layers")
    parser.add_argument('-ud', '--num-discrim-update', type=int, default=1, help="Discriminator updates per iteration")
    parser.add_argument('-ug', '--num-gen-update', type=int, default=6, help="Generator updates per iteration")
    parser.add_argument('--policy-lr', type=float, default=5e-5, help="Policy network learning rate")
    parser.add_argument('--value-lr', type=float, default=1e-4, help="Value network learning rate")
    parser.add_argument('--disc-lr', type=float, default=5e-5, help="Discriminator learning rate")
    parser.add_argument('--eps', type=float, default=1e-8, help="Epsilon for numerical stability")
    parser.add_argument('-g', '--gamma', type=float, default=0.98, help="Discount factor")
    parser.add_argument('--cuda', type=bool, default=True, help="Use CUDA if available")
    parser.add_argument('--data', type=str, default="Data/clean_DJI.pkl", help="Path to environment data file")
    parser.add_argument('--cjson', type=str, default="Data/C.json", help="Path to context JSON")
    parser.add_argument('--no-ppo', action='store_true', help="Disable PPO (use vanilla policy gradient)")
    parser.add_argument('--no-wgan-gp', action='store_true', help="Disable WGAN-GP (use vanilla GAN loss)")
    parser.add_argument('--vanilla', action='store_true', help="Classic GAIL (no PPO, no WGAN-GP)")
    parser.add_argument('--exp-name', type=str, default=None, help="Experiment folder name (auto if None)")
    return parser.parse_args()


def pad_stack(lst, device, dtype=torch.float32):
    if not lst:
        return torch.empty(0, device=device, dtype=dtype)
    lens = [x.shape[0] for x in lst]
    T_max = max(lens) if lens else 0
    if T_max == 0:
        return torch.empty(0, device=device, dtype=dtype)
    out_list = []
    for x in lst:
        t = torch.tensor(x, dtype=dtype, device=device)
        if t.size(0) < T_max:
            pad = torch.zeros((T_max - t.size(0), *t.shape[1:]), device=device, dtype=dtype)
            t = torch.cat([t, pad], dim=0)
        out_list.append(t)
    return torch.stack(out_list, dim=0)


def collect_trajectories(env, policy, num_trajs, max_steps, device):
    print("Starting trajectory collection...")
    S, C, A, R, D, L, Old_logp = [], [], [], [], [], [], []
    
    for epi in range(num_trajs):
        t0 = time.time()
        state = env.reset()
        if state is None or np.any(np.isnan(state)):
            print(f"[Episode {epi+1}] Invalid initial state → skip")
            continue

        states = [state]
        ctxs, acts, rews, dones, old_logps = [], [], [], [], []
        hidden = None

        for t in range(max_steps):
            v_ctx, l_ctx, e_ctx = env._get_context()
            context = np.concatenate([v_ctx, l_ctx, e_ctx], axis=0)
            ctxs.append(context)
            
            s_t = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            c_t = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            seq_len = torch.tensor([1], dtype=torch.long).to(device)
            
            with torch.no_grad():
                action, mu_sel, log_s_sel, comp, hidden = policy.sample(s_t, c_t, seq_len, hidden=hidden)
                dist = torch.distributions.Normal(mu_sel, torch.exp(log_s_sel) + 1e-8)
                log_prob = dist.log_prob(action).sum(dim=-1).item()
                old_logps.append(log_prob)

            action = action[0, 0].cpu().numpy()
            step_result = env.step(action)
            last_s = step_result.cur_state
            act = step_result.action
            nxt_s = step_result.next_state
            r = step_result.reward
            done = step_result.done

            if np.any(np.isnan(nxt_s)):
                print(f"[Episode {epi+1}, Step {t+1}] NaN in next state → break")
                break

            states.append(nxt_s)
            acts.append(action)
            rews.append(r)
            dones.append(done)

            if done:
                break
        
        if len(acts) < 1:
            print(f"[Episode {epi+1}] No actions collected → skip")
            continue

        T = len(acts)
        S.append(np.stack(states[:T]))
        C.append(np.stack(ctxs[:T]))
        A.append(np.stack(acts[:T]))
        R.append(np.array(rews[:T]))
        D.append(np.array(dones[:T]))
        L.append(T)
        Old_logp.append(np.array(old_logps[:T]))
        
        elapsed = time.time() - t0
        if epi == 0 or (epi + 1) % 10 == 0:
            print(f"Collected {epi+1}/{num_trajs} episodes, len={T}, time={elapsed:.2f}s, reward_mean={np.mean(rews):.4f}")
    
    print("Trajectory collection completed.")
    return S, C, A, R, D, L, Old_logp


def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'rnn' in name.lower():
                init.xavier_uniform_(param)
            elif 'linear' in name.lower():
                init.xavier_uniform_(param)
        elif 'bias' in name:
            init.zeros_(param)


def main(args):
    # Parse ablation options and auto-generate experiment folder name
    ablation_tags = []
    if args.vanilla:
        ablation_tags.append('vanilla')
        args.no_ppo = True
        args.no_wgan_gp = True
    else:
        if args.no_ppo:
            ablation_tags.append('noPPO')
        if args.no_wgan_gp:
            ablation_tags.append('noWGAN-GP')
        if not ablation_tags:
            ablation_tags.append('full')

    if args.exp_name:
        exp_folder = os.path.join("pth", f"pth_{args.exp_name}")
    else:
        exp_folder = os.path.join("pth", f"pth_{'_'.join(ablation_tags)}")

    os.makedirs(exp_folder, exist_ok=True)
    log_path = os.path.join(exp_folder, "training_log.txt")

    print("Initializing environment and models...")
    env = MicroTrajectoryEnv(args.data, c_json_path=args.cjson, history_length=5, total_iterations=args.iteration)
    expert_trajs = env.expert_trajs
    
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
    policy = PolicyNetRNN(
        state_dim=6, context_dim=39, hidden_dim=args.hidden, num_layers=args.num_layers, K=5,
        action_scale=torch.tensor([0.01, 0.01, 0.5, 0.5, 1.5, 1.5])
    ).to(device)
    old_policy = PolicyNetRNN(
        state_dim=6, context_dim=39, hidden_dim=args.hidden, num_layers=args.num_layers, K=5,
        action_scale=torch.tensor([0.01, 0.01, 0.5, 0.5, 1.5, 1.5])
    ).to(device)
    value = ValueNetRNN(state_dim=6, context_dim=39, hidden_dim=args.hidden, num_layers=args.num_layers).to(device)
    disc = DiscriminatorRNN(state_dim=6, context_dim=39, action_dim=6, hidden_dim=args.hidden, num_layers=args.num_layers).to(device)
    
    initialize_weights(policy)
    initialize_weights(old_policy)
    initialize_weights(value)
    initialize_weights(disc)
    
    opt_pi = optim.Adam(policy.parameters(), lr=args.policy_lr, eps=args.eps, amsgrad=True)
    opt_v = optim.Adam(value.parameters(), lr=args.value_lr, eps=args.eps, amsgrad=True)
    opt_d = optim.Adam(disc.parameters(), lr=args.disc_lr, eps=args.eps, amsgrad=True)
    
    eff_bs = max(args.batch_size // 2, 2)
    
    # Prepare expert batches
    expert_S, expert_C, expert_A, expert_L = [], [], [], []
    for traj in expert_trajs[:args.n_episode]:
        T = traj["seq_len"] - 1
        if T < 1:
            continue
        states = traj["states"][:T].copy()
        states[:, 2:4] += np.random.normal(0, 0.01, states[:, 2:4].shape)
        expert_S.append(states)
        expert_C.append(np.hstack([traj["vehicle_context"][:T],
                                   traj["lane_context"][:T],
                                   traj["exists_context"][:T]]))
        actions = np.diff(traj["states"][:T+1], axis=0)
        if len(actions) != T:
            print(f"Warning: actions length {len(actions)} does not match T {T}")
            continue
        expert_A.append(actions)
        expert_L.append(T)
    
    if not expert_S:
        print("No valid expert trajectories available. Exiting...")
        return
    
    expert_S_t = pad_stack(expert_S, device)
    expert_C_t = pad_stack(expert_C, device)
    expert_A_t = pad_stack(expert_A, device)
    expert_L_t = torch.tensor(expert_L, dtype=torch.long, device=device)
    
    for it in range(args.iteration):
        t_start = time.time()
        print(f"\n===== Iteration {it+1}/{args.iteration} =====")
        
        env.set_current_iteration(it)
        old_policy.load_state_dict(policy.state_dict())
        
        S, C, A, R, D, Ls, Old_logp = collect_trajectories(env, old_policy, args.n_episode, args.max_steps, device)
        if not S:
            print("No valid trajectories collected → skip iteration")
            continue
        
        rollout_n = len(S)
        num_batches = (rollout_n + eff_bs - 1) // eff_bs
        
        S_t = pad_stack(S, device)
        C_t = pad_stack(C, device)
        A_t = pad_stack(A, device)
        L_t = torch.tensor(Ls, dtype=torch.long, device=device)
        Old_logp_t = pad_stack([logp.reshape(-1, 1) for logp in Old_logp], device)
        
        # ---------------------------
        # 1) Discriminator update
        # ---------------------------
        disc.train()
        disc_loss = 0.0
        d_gen_mean = 0.0
        d_exp_mean = 0.0
        print("Updating Discriminator...")
        
        for _ in range(args.num_discrim_update):
            opt_d.zero_grad()
            batch_loss = 0.0
            for b in range(num_batches):
                start = b * eff_bs
                end = min((b + 1) * eff_bs, rollout_n)
                s_b = S_t[start:end]
                c_b = C_t[start:end]
                a_b = A_t[start:end]
                l_b = L_t[start:end]
                if s_b.size(0) == 0:
                    continue
                
                s_e = expert_S_t[start:end]
                c_e = expert_C_t[start:end]
                a_e = expert_A_t[start:end]
                l_e = expert_L_t[start:end]
                
                T_min = min(s_b.size(1), s_e.size(1), c_b.size(1), c_e.size(1), a_b.size(1), a_e.size(1))
                if T_min < 1:
                    continue
                s_b = s_b[:, :T_min]; c_b = c_b[:, :T_min]; a_b = a_b[:, :T_min]
                s_e = s_e[:, :T_min]; c_e = c_e[:, :T_min]; a_e = a_e[:, :T_min]
                l_b = torch.clamp(l_b, min=1, max=T_min)
                l_e = torch.clamp(l_e, min=1, max=T_min)
                
                d_gen = disc(s_b, c_b, a_b, l_b)
                d_exp = disc(s_e, c_e, a_e, l_e)
                
                if not args.no_wgan_gp:
                    # Gradient penalty (fixed factor 0.5; no CLI arg used)
                    gp = disc.gradient_penalty((s_e, c_e, a_e), (s_b, c_b, a_b), l_b, device=device) * 0.5
                    loss_d = (d_gen.mean() - d_exp.mean()) + gp
                    loss_d = loss_d / num_batches
                else:
                    bce = torch.nn.BCEWithLogitsLoss()
                    real_labels = torch.ones_like(d_exp)
                    fake_labels = torch.zeros_like(d_gen)
                    loss_d = (bce(d_exp, real_labels) + bce(d_gen, fake_labels)) / num_batches

                loss_d.backward()
                batch_loss += loss_d.item()
                d_gen_mean += d_gen.mean().item()
                d_exp_mean += d_exp.mean().item()
            
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 0.5)
            opt_d.step()
            disc_loss += batch_loss
            torch.cuda.empty_cache()
        
        d_gen_mean /= num_batches
        d_exp_mean /= num_batches
        print(f"D(gen) mean: {d_gen_mean:.4f}, D(exp) mean: {d_exp_mean:.4f}")
        
        # ---------------------------
        # 2) Compute rewards via D
        # ---------------------------
        disc.eval()
        print("Calculating rewards...")
        with torch.no_grad():
            rew_list = []
            for b in range(num_batches):
                start = b * eff_bs
                end = min((b + 1) * eff_bs, rollout_n)
                s_b = S_t[start:end]
                c_b = C_t[start:end]
                a_b = A_t[start:end]
                l_b = L_t[start:end]
                if s_b.size(0) == 0:
                    continue
                
                T_min = min(s_b.size(1), a_b.size(1), c_b.size(1))
                if T_min < 1:
                    continue
                s_b = s_b[:, :T_min]; c_b = c_b[:, :T_min]; a_b = a_b[:, :T_min]
                l_b = torch.clamp(l_b, min=1, max=T_min)
                
                r_b = -disc(s_b, c_b, a_b, l_b)
                for i in range(s_b.size(0)):
                    Li = l_b[i].item()
                    rew_list.append(r_b[i].expand(Li).cpu())
            rew_all = torch.cat(rew_list, dim=0) if rew_list else torch.zeros(1, device=device)
            print("Reward stats before scaling:", rew_all.min().item(), rew_all.max().item())
            r_min, r_max = rew_all.min(), rew_all.max()
            rew_all = (rew_all - r_min) / (r_max - r_min + 1e-6)
            print("Reward stats after scaling:", rew_all.min().item(), rew_all.max().item())
        
        # ---------------------------
        # 3) Update Policy & Value
        # ---------------------------
        print("Updating Policy & Value networks...")
        policy.train()
        value.train()
        policy_loss, value_loss = 0.0, 0.0
        
        for gen_upd in range(args.num_gen_update):
            opt_pi.zero_grad()
            opt_v.zero_grad()
            
            tot_ptr = 0
            for b in range(num_batches):
                start = b * eff_bs
                end = min((b + 1) * eff_bs, rollout_n)
                s_b = S_t[start:end]
                c_b = C_t[start:end]
                a_b = A_t[start:end]
                l_b = L_t[start:end]
                old_logp_b = Old_logp_t[start:end]
                if s_b.size(0) == 0:
                    continue
                
                vs_list, new_logp_list, r_list, ep_lengths, old_lps_list = [], [], [], [], []
                for i in range(s_b.size(0)):
                    Li = l_b[i].item()
                    if Li < 1:
                        continue
                    ss = s_b[i, :Li].unsqueeze(0)
                    cc = c_b[i, :Li].unsqueeze(0)
                    aa = a_b[i, :Li].unsqueeze(0)
                    seq_len = torch.tensor([Li], dtype=torch.long, device=device)
                    
                    v_out, _ = value(ss, cc, seq_len)
                    v_flat = v_out.squeeze(-1).reshape(-1)
                    
                    mu, log_s, pi, _ = policy(ss, cc, seq_len)
                    act = aa.unsqueeze(2)
                    dist = torch.distributions.Normal(mu, torch.exp(log_s) + args.eps)
                    new_log_p_permix = dist.log_prob(act).sum(dim=-1)
                    new_log_p = torch.logsumexp(
                        new_log_p_permix + torch.log(pi.clamp(min=1e-6) + args.eps), dim=2
                    ).reshape(-1)
                    
                    if torch.any(torch.isnan(new_log_p)):
                        continue
                    
                    old_logp_i = old_logp_b[i, :Li, 0]
                    
                    vs_list.append(v_flat)
                    new_logp_list.append(new_log_p)
                    old_lps_list.append(old_logp_i)
                    if tot_ptr + Li <= rew_all.size(0):
                        r_list.append(rew_all[tot_ptr:tot_ptr+Li])
                    else:
                        continue
                    ep_lengths.append(Li)
                    tot_ptr += Li
                
                if not ep_lengths:
                    continue
                
                vs_ = torch.cat(vs_list, dim=0)
                new_lps_ = torch.cat(new_logp_list, dim=0)
                old_lps_ = torch.cat(old_lps_list, dim=0)
                r_all = torch.cat(r_list, dim=0)
                
                if r_all.size(0) == 0:
                    continue
                
                r_all = torch.clamp(r_all, min=-1e6, max=1e6)
                vs_ = torch.clamp(vs_, min=-1e6, max=1e6)
                
                R_, Adv_ = [], []
                start_ptr = 0
                for Li in ep_lengths:
                    r_ep = r_all[start_ptr:start_ptr+Li].cpu().numpy()
                    v_ep = vs_[start_ptr:start_ptr+Li].detach().cpu().numpy()
                    if len(r_ep) == 0 or len(v_ep) == 0:
                        start_ptr += Li
                        continue
                    R_ep, Adv_ep = get_gae(r_ep, np.array([Li]), v_ep, args.gamma, lamda=0.98, device=device)
                    if len(R_ep) > 0 and len(Adv_ep) > 0:
                        R_.append(R_ep.to(device) if isinstance(R_ep, torch.Tensor) else torch.from_numpy(R_ep).to(device))
                        Adv_.append(Adv_ep.to(device) if isinstance(Adv_ep, torch.Tensor) else torch.from_numpy(Adv_ep).to(device))
                    start_ptr += Li
                
                if not Adv_:
                    continue
                
                R_ = torch.cat(R_, dim=0)
                Adv_ = torch.cat(Adv_, dim=0)
                
                diff = new_lps_ - old_lps_
                diff = torch.clamp(diff, min=-20, max=20)
                ratio = torch.clamp(torch.exp(diff), 0.8, 1.2)

                # Optional diagnostics
                if (it + 1) % 5 == 0 or it == 0:
                    print(f"Iter {it+1}, Batch {b+1}: diff min={diff.min().item():.4f}, max={diff.max().item():.4f}, mean={diff.mean().item():.4f}")
                    print(f"Iter {it+1}, Batch {b+1}: ratio min={ratio.min().item():.4f}, max={ratio.max().item():.4f}, mean={ratio.mean().item():.4f}")

                if torch.any(torch.isnan(ratio)) or ratio.size(0) != Adv_.size(0):
                    continue

                if not args.no_ppo:
                    # PPO clipped objective
                    surr1 = ratio * Adv_
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * Adv_
                    loss_pi = -torch.min(surr1, surr2).mean()
                    entropy = dist.entropy().mean()
                    loss_pi = loss_pi + 0.01 * entropy
                else:
                    # Vanilla policy gradient
                    loss_pi = -(new_lps_ * Adv_).mean()
                    entropy = dist.entropy().mean()
                    loss_pi = loss_pi + 0.01 * entropy
                
                loss_v = (R_ - vs_).pow(2).mean()
                
                loss_pi /= num_batches
                loss_v /= num_batches
                
                loss_pi.backward()
                loss_v.backward()
                policy_loss += loss_pi.item()
                value_loss += loss_v.item()
            
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(value.parameters(), 0.1)
            opt_pi.step()
            opt_v.step()
            torch.cuda.empty_cache()
        
        avg_rew = rew_all.mean().item() if rew_all.numel() > 0 else 0.0
        avg_len = L_t.float().mean().item() if L_t.numel() > 0 else 0
        elapsed = time.time() - t_start
        
        policy_loss /= args.num_gen_update
        value_loss /= args.num_gen_update
        
        log_str = (f"Iter {it+1} | Reward = {avg_rew:.4f}, Length = {avg_len:.2f}, "
                   f"Discrim Loss = {disc_loss:.5f}, Policy Loss = {policy_loss:.5f}, "
                   f"Value Loss = {value_loss:.5f}, Time = {elapsed:.2f}s")
        print(log_str)
        
        os.makedirs("pth", exist_ok=True)
        with open(log_path, "a") as f:
            f.write(log_str + "\n")
        
        if (it + 1) % 10 == 0 or it == 0 or it == args.iteration - 1:
            torch.save(policy.state_dict(), os.path.join(exp_folder, f"policy_net_iter{it+1}.pth"))

    print("Training completed.")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = argparser()
    print("Arguments:", args)
    main(args)
