import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from shapely.geometry import Polygon, Point
import json
from utils.utils import Step

class MicroTrajectoryEnv(object):
    """
    Environment built on clean_DJI.pkl to generate per-vehicle micro trajectories.
    Provides data normalization, trajectory extraction with padding, state transition,
    reward calculation, and history-based belief state construction.

    Notes:
    - All center_x, center_y (including neighbors) are normalized to [0,1] by dividing by 3840 and 2160.
    - Distances → MinMaxScaler, velocities/accelerations → StandardScaler.
    - Trajectories are split into environment vs. expert sets (70:30) with a fixed seed.
    """

    def __init__(self, pkl_path="Data/clean_DJI.pkl", c_json_path="Data/C.json",
                 history_length=5, expert_split_ratio=0.3, seed=42, total_iterations=100):
        self.pkl_path = pkl_path
        self.c_json_path = c_json_path
        self.expert_split_ratio = expert_split_ratio
        self.seed = seed

        # 1) Load & filter data by per-ID length (5–95 percentile)
        self.data = pd.read_pickle(pkl_path)
        frame_counts = self.data["ID"].value_counts()
        lower, upper = np.percentile(frame_counts, [5, 95])
        valid_ids = frame_counts[(frame_counts >= lower) & (frame_counts <= upper)].index
        self.data = self.data[self.data["ID"].isin(valid_ids)].reset_index(drop=True)

        # Logging filtered counts
        total_ids = len(frame_counts)
        filtered_ids = len(valid_ids)
        print(f"[INFO] Total IDs: {total_ids}, Filtered IDs: {filtered_ids}, Excluded IDs: {total_ids - filtered_ids}")

        # 2) Column groups
        self.state_cols = ["center_x", "center_y", "v_x", "v_y", "a_x", "a_y"]
        self.standard_cols = [
            'v_x', 'v_y', 'a_x', 'a_y',
            'preceding_rel_v_x', 'preceding_rel_v_y',
            'following_rel_v_x', 'following_rel_v_y',
            'left_preceding_rel_v_x', 'left_preceding_rel_v_y',
            'right_preceding_rel_v_x', 'right_preceding_rel_v_y',
            'left_following_rel_v_x', 'left_following_rel_v_y',
            'right_following_rel_v_x', 'right_following_rel_v_y'
        ]
        self.minmax_cols = [
            'preceding_distance', 'following_distance',
            'left_preceding_distance', 'right_preceding_distance',
            'left_following_distance', 'right_following_distance'
        ]
        self.vehicle_context_cols = [
            "preceding_distance", "right_preceding_distance", "left_preceding_distance",
            "following_distance", "right_following_distance", "left_following_distance",
            "preceding_rel_v_x", "preceding_rel_v_y",
            "right_preceding_rel_v_x", "right_preceding_rel_v_y",
            "left_preceding_rel_v_x", "left_preceding_rel_v_y",
            "following_rel_v_x", "following_rel_v_y",
            "right_following_rel_v_x", "right_following_rel_v_y",
            "left_following_rel_v_x", "left_following_rel_v_y",
            "preceding_center_x", "preceding_center_y",
            "right_preceding_center_x", "right_preceding_center_y",
            "left_preceding_center_x", "left_preceding_center_y",
            "following_center_x", "following_center_y",
            "right_following_center_x", "right_following_center_y",
            "left_following_center_x", "left_following_center_y"
        ]
        self.exists_context_cols = [
            "preceding_exists", "right_preceding_exists", "left_preceding_exists",
            "following_exists", "right_following_exists", "left_following_exists"
        ]

        # 3) Max sequence length (clipped to [200, 2000])
        self.max_seq_length = int(np.percentile(frame_counts, 90))
        self.max_seq_length = max(200, min(self.max_seq_length, 2000))

        # 4) Read C.json and normalize lane polygons to [0,1]
        with open(c_json_path, "r", encoding="utf-8") as f:
            self.C_json = json.load(f)
        self.video_id = list(self.C_json.keys())[0]
        self.lane_data = self.C_json[self.video_id]["A"]
        width, height = 3840.0, 2160.0
        for ln in ("1", "2", "3"):
            coords = self.lane_data[ln]
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])  # ensure closed polygon
            self.lane_data[ln] = [[x / width, y / height] for x, y in coords]

        # 5) Normalize data columns
        self._normalize_data()

        # 6) Lane polygons (on [0,1] coordinates)
        self.lane_poly = {ln: Polygon(self.lane_data[ln]) for ln in ("1", "2", "3")}

        # 6-B) Value clipping ranges for state vars (percentiles from data)
        self.v_x_lower, self.v_x_upper = np.percentile(self.data["v_x"], [5, 95])
        self.v_y_lower, self.v_y_upper = np.percentile(self.data["v_y"], [5, 95])
        self.a_x_lower, self.a_x_upper = np.percentile(self.data["a_x"], [5, 95])
        self.a_y_lower, self.a_y_upper = np.percentile(self.data["a_y"], [5, 95])

        # 7) Initial state distribution
        self.initial_states = []
        for _, group in self.data.groupby("ID"):
            group = group.sort_values("frame_number")
            init_st = group[self.state_cols].iloc[0].values
            self.initial_states.append(init_st)
        self.initial_states = np.array(self.initial_states)

        # 8) Split trajectories: environment vs. expert
        self._split_and_load_trajectories()

        # 9) History buffers
        self.history_length = history_length
        self.state_history = []
        self.context_history = []

        # 10) Training progress trackers (for dynamic scaling in transition/reward)
        self.current_iteration = 0
        self.total_iterations = total_iterations

    def set_current_iteration(self, iteration):
        """Set the current training iteration (used for dynamic scaling)."""
        self.current_iteration = iteration

    def _normalize_data(self):
        """
        1) Any '*center_x' → x/3840 clipped to [0,1]; any '*center_y' → y/2160 clipped to [0,1]
        2) Distances (…distance) → MinMaxScaler (with 1–99 percentile clipping)
        3) Vel/Acc (standard_cols) → StandardScaler, then clip to [-1.5, 1.5]
        """
        width = 3840.0
        height = 2160.0
        all_cols = list(self.data.columns)
        for col in all_cols:
            if "center_x" in col:
                self.data[col] = np.clip(self.data[col] / width, 0.0, 1.0)
            elif "center_y" in col:
                self.data[col] = np.clip(self.data[col] / height, 0.0, 1.0)

        self.minmax_scaler = MinMaxScaler()
        if self.minmax_cols:
            for c in self.minmax_cols:
                lo, hi = np.percentile(self.data[c], [1, 99])
                self.data[c] = np.clip(self.data[c], lo, hi)
            self.data[self.minmax_cols] = self.minmax_scaler.fit_transform(self.data[self.minmax_cols])

        self.standard_scaler = StandardScaler()
        if self.standard_cols:
            self.data[self.standard_cols] = self.standard_scaler.fit_transform(self.data[self.standard_cols])
            self.data[self.standard_cols] = np.clip(self.data[self.standard_cols], -1.5, 1.5)

    def _split_and_load_trajectories(self):
        """Split into env trajectories (self.trajs) and expert trajectories (self.expert_trajs) by ratio."""
        grouped = [group for _, group in self.data.groupby("ID")]
        np.random.seed(self.seed)
        np.random.shuffle(grouped)

        total_groups = len(grouped)
        expert_count = max(1, int(total_groups * self.expert_split_ratio))
        env_count = total_groups - expert_count

        env_groups = grouped[:env_count]
        expert_groups = grouped[env_count:]

        self.trajs = self._create_trajectories(env_groups)
        self.expert_trajs = self._create_trajectories(expert_groups)

        self.initial_states = []
        for traj in self.trajs:
            self.initial_states.append(traj["states"][0])
        self.initial_states = np.array(self.initial_states) if self.initial_states else np.array([])

        self.current_traj_idx = 0
        self.current_step = 0
        self.target_len = 0

    def _create_trajectories(self, groups):
        trajs = []
        for grp in groups:
            grp = grp.sort_values("frame_number")
            states = grp[self.state_cols].values
            vehicle_ctx = grp[self.vehicle_context_cols].values

            # Lane context: one-hot by lane polygon hit; 0-vector if not found
            lane_ctx = []
            for s in states:
                found = self._locate_lane_roi(s[0], s[1])
                lane_ctx.append([0, 0, 0] if found is None else np.eye(3)[found])
            lane_ctx = np.array(lane_ctx, dtype=float)

            exists_ctx = grp[self.exists_context_cols].values
            seq_len = len(states)
            if seq_len < self.max_seq_length:
                pad_len = self.max_seq_length - seq_len
                states = np.pad(states, ((0, pad_len), (0, 0)), constant_values=0)
                vehicle_ctx = np.pad(vehicle_ctx, ((0, pad_len), (0, 0)), constant_values=0)
                lane_ctx = np.pad(lane_ctx, ((0, pad_len), (0, 0)), constant_values=0)
                exists_ctx = np.pad(exists_ctx, ((0, pad_len), (0, 0)), constant_values=0)
                mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])
            else:
                states = states[:self.max_seq_length]
                vehicle_ctx = vehicle_ctx[:self.max_seq_length]
                lane_ctx = lane_ctx[:self.max_seq_length]
                exists_ctx = exists_ctx[:self.max_seq_length]
                mask = np.ones(self.max_seq_length)

            episode = []
            for i in range(len(states) - 1):
                cur = states[i]
                nxt = states[i + 1]
                act = nxt - cur
                done = (i == len(states) - 2)
                next_ctx = np.concatenate([vehicle_ctx[i+1], lane_ctx[i+1], exists_ctx[i+1]])
                episode.append(Step(cur_state=cur, action=act, next_state=nxt,
                                    next_context=next_ctx, reward=0, done=done, mask=mask[i+1]))

            trajs.append(dict(
                states=states,
                vehicle_context=vehicle_ctx,
                lane_context=lane_ctx,
                exists_context=exists_ctx,
                mask=mask,
                seq_len=min(seq_len, self.max_seq_length),
                episode=episode
            ))
        return trajs

    def _locate_lane_roi(self, x, y):
        """Return lane index {0,1,2} if (x,y) hits polygon, else None. Coordinates are in [0,1]."""
        pt = Point(x, y)
        for i, ln in enumerate(["1", "2", "3"]):
            if self.lane_poly[ln].contains(pt):
                return i
        return None

    def get_reward(self, state, vehicle_context=None, lane_context=None, exists_context=None):
        if vehicle_context is None or lane_context is None or exists_context is None:
            vehicle_context, lane_context, exists_context = self._get_context()

        # Unpack state
        center_x, center_y, v_x, v_y, a_x, a_y = state[0], state[1], state[2], state[3], state[4], state[5]

        # Heuristics based on observed data stats (values from pre-analysis)
        v_x_min, v_x_max = -1.717, 2.200
        a_x_min, a_x_max = -3.160, 2.969
        v_x_mean, v_x_std = -0.009, 0.770
        a_x_mean, a_x_std = -0.001, 0.758

        # 1) Speed reward
        speed_reward = 0.0
        if v_x < 0:  # desired driving direction (left)
            v_x_z = (v_x - v_x_mean) / v_x_std
            if -1.0 <= v_x_z <= 1.0:
                speed_reward = 0.3
            elif v_x_min <= v_x <= v_x_max:
                speed_reward = 0.1
            else:
                speed_reward = -0.1
        else:
            speed_reward = -0.2  # wrong direction

        # v_y close to 0 is better (stability)
        if abs(v_y) <= 1.0:  # after StandardScaler, assume |z| ≤ 1
            speed_reward += 0.2
        else:
            speed_reward -= 0.1

        # 2) Acceleration reward
        accel_reward = 0.0
        a_x_z = (a_x - a_x_mean) / a_x_std
        if abs(a_x_z) <= 0.5:
            accel_reward = 0.2
        elif a_x_min <= a_x <= a_x_max:
            accel_reward = 0.0
        else:
            accel_reward = -0.1

        # a_y close to 0 is preferred
        if abs(a_y) <= 1.0:
            accel_reward += 0.1
        else:
            accel_reward -= 0.1

        # 3) Direction reward (x decreasing over time)
        direction_reward = 0.0
        if len(self.state_history) >= self.history_length:
            prev_x = self.state_history[-2][0]
            delta_x = prev_x - center_x
            if delta_x > 0.01:      # drop >1% in [0,1] scale
                direction_reward = 0.3
            elif delta_x > 0:
                direction_reward = 0.1
            else:
                direction_reward = -0.2

        # 4) Lane reward (change vs. keep)
        lane_reward = 0.0
        if len(self.state_history) > 1:
            prev_lane = self.context_history[-2][1]
            changed = not np.array_equal(prev_lane, lane_context)
            if changed:
                min_dist = 0.15
                dists = vehicle_context[:6]
                exist = exists_context[:6]
                safe = all(d >= min_dist for d, e in zip(dists, exist) if e)
                lane_reward = 0.5 if safe else -0.3
            else:
                found_idx = self._locate_lane_roi(center_x, center_y)
                lane_idx = np.argmax(lane_context)
                if found_idx == lane_idx:
                    lane_reward = 0.4 if lane_idx == 2 else 0.1
                else:
                    lane_reward = -0.1
        else:
            found_idx = self._locate_lane_roi(center_x, center_y)
            lane_idx = np.argmax(lane_context)
            if found_idx == lane_idx:
                lane_reward = 0.4 if lane_idx == 2 else 0.1
            else:
                lane_reward = -0.1

        # 5) Safety distance reward
        safety_reward = 0.0
        min_dist = 0.1
        dist_6 = vehicle_context[:6]
        exist_6 = exists_context[:6]
        if any(exist_6):
            min_d = min(d for d, e in zip(dist_6, exist_6) if e)
            if min_d >= min_dist:
                safety_reward = 0.1
            else:
                safety_reward = -0.2 * (min_dist - min_d)
        else:
            safety_reward = 0.0

        # 6) Similarity penalty to expert state (annealed by training progress)
        expert_state = self.trajs[self.current_traj_idx]["states"][self.current_step]
        mse = np.mean((state - expert_state) ** 2)
        penalty_weight = 0.005 if self.total_iterations == 0 else 0.005 * (1 - self.current_iteration / self.total_iterations)
        similarity_penalty = mse * penalty_weight  # kept local (no attribute stored)

        # 7) Total reward (scaled and clipped)
        total_reward = speed_reward + accel_reward + direction_reward + lane_reward + safety_reward - similarity_penalty
        total_reward *= 10.0
        total_reward = np.clip(total_reward, -10.0, 10.0)
        return total_reward

    def get_state_transition(self, state, action):
        traj = self.trajs[self.current_traj_idx]
        target_next = traj["states"][min(self.current_step + 1, traj["seq_len"] - 1)]

        # Clip per-dimension action deltas
        action[0] = np.clip(action[0], -0.05, 0.05)   # center_x
        action[1] = np.clip(action[1], -0.05, 0.05)   # center_y
        action[2] = np.clip(action[2], -0.5, 0.5)     # v_x
        action[3] = np.clip(action[3], -0.5, 0.5)     # v_y
        action[4] = np.clip(action[4], -1.0, 1.0)     # a_x
        action[5] = np.clip(action[5], -1.0, 1.0)     # a_y

        diff = target_next - state
        scale_factor = self.get_scale_factor()  # anneal from 0.3 → 0
        nxt_s = state + (scale_factor * diff) + ((1 - scale_factor) * action)

        # Clip state ranges
        nxt_s[0] = np.clip(nxt_s[0], 0.0, 1.0)                       # center_x
        nxt_s[1] = np.clip(nxt_s[1], 0.0, 1.0)                       # center_y
        nxt_s[2] = np.clip(nxt_s[2], self.v_x_lower, self.v_x_upper) # v_x
        nxt_s[3] = np.clip(nxt_s[3], self.v_y_lower, self.v_y_upper) # v_y
        nxt_s[4] = np.clip(nxt_s[4], self.a_x_lower, self.a_x_upper) # a_x
        nxt_s[5] = np.clip(nxt_s[5], self.a_y_lower, self.a_y_upper) # a_y
        return nxt_s
    
    def get_scale_factor(self):
        """Return current scale factor used in state transition blending."""
        if self.total_iterations == 0:
            return 0.0
        return max(0.0, 0.3 - (self.current_iteration / self.total_iterations) * 0.3)

    def reset(self, start_state=None):
        if start_state is None:
            cx_bins = np.percentile(self.initial_states[:, 0], [0, 25, 50, 75, 100])
            bin_probs = [0.15, 0.25, 0.35, 0.25]
            lane_probs = [0.457, 0.435, 0.108]
            max_attempts = 3
            attempt = 0
            valid_indices = []

            # Try matching both x-bin and lane at t=0
            while attempt < max_attempts:
                bidx = np.random.choice(4, p=bin_probs)
                lo, hi = cx_bins[bidx], cx_bins[bidx + 1]
                lane_idx = np.random.choice([0, 1, 2], p=lane_probs)
                cand = []
                for i, st in enumerate(self.initial_states):
                    if lo <= st[0] <= hi and np.argmax(self.trajs[i]["lane_context"][0]) == lane_idx:
                        cand.append(i)
                if cand:
                    valid_indices = cand
                    break
                attempt += 1

            # Fallback: match only x-bin
            if not valid_indices:
                attempt = 0
                while attempt < max_attempts:
                    bidx = np.random.choice(4, p=bin_probs)
                    lo, hi = cx_bins[bidx], cx_bins[bidx + 1]
                    cand = [i for i, st in enumerate(self.initial_states) if lo <= st[0] <= hi]
                    if cand:
                        valid_indices = cand
                        break
                    attempt += 1

            if not valid_indices:
                valid_indices = list(range(len(self.initial_states)))

            idx = np.random.choice(valid_indices)
            start_state = self.initial_states[idx].copy()
            self.current_traj_idx = idx
        else:
            # If a start_state is provided, keep current behavior: pick a random trajectory
            self.current_traj_idx = np.random.randint(len(self.trajs))

        self.current_step = 0
        self.target_len = self.trajs[self.current_traj_idx]["seq_len"]
        self.state_history = []
        self.context_history = []

        st_arr = self.trajs[self.current_traj_idx]["states"]
        veh_arr = self.trajs[self.current_traj_idx]["vehicle_context"]
        lan_arr = self.trajs[self.current_traj_idx]["lane_context"]
        ex_arr = self.trajs[self.current_traj_idx]["exists_context"]

        st_idx = max(0, self.current_step - self.history_length + 1)
        for i in range(st_idx, self.current_step + 1):
            if i >= 0:
                self.state_history.append(st_arr[i])
                self.context_history.append((veh_arr[i], lan_arr[i], ex_arr[i]))
            else:
                self.state_history.append(np.zeros_like(start_state))
                self.context_history.append((
                    np.zeros_like(veh_arr[0]),
                    np.zeros_like(lan_arr[0]),
                    np.zeros_like(ex_arr[0])
                ))

        self.state_history[-1] = start_state
        self._cur_state = start_state
        self._is_done = False
        return self._cur_state

    def step(self, action):
        # Terminal guard
        if self._is_done:
            veh, lan, ex = self._get_context()
            next_context = np.concatenate([veh, lan, ex])
            mask_index = min(self.current_step, self.target_len - 1)
            return Step(
                cur_state=self._cur_state,
                action=action,
                next_state=self._cur_state,
                next_context=next_context,
                reward=self.get_reward(self._cur_state, veh, lan, ex),
                done=True,
                mask=self.trajs[self.current_traj_idx]["mask"][mask_index]
            )

        # Normal transition
        last_s = self._cur_state
        nxt_s = self.get_state_transition(last_s, action)
        veh, lan, ex = self._get_context()
        next_context = np.concatenate([veh, lan, ex])
        r = self.get_reward(nxt_s, veh, lan, ex)

        self._cur_state = nxt_s
        self.current_step += 1
        self.state_history.append(self._cur_state)
        self.context_history.append((veh, lan, ex))

        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)
            self.context_history.pop(0)

        # Check terminal
        self._is_done = (self.current_step >= self.target_len)
        mask_index = min(self.current_step, self.target_len - 1)
        return Step(
            cur_state=last_s,
            action=action,
            next_state=nxt_s,
            next_context=next_context,
            reward=r,
            done=self._is_done,
            mask=self.trajs[self.current_traj_idx]["mask"][mask_index]
        )

    def _get_context(self):
        traj = self.trajs[self.current_traj_idx]
        if self.current_step < traj["states"].shape[0]:
            vc = traj["vehicle_context"][self.current_step]
            lc = traj["lane_context"][self.current_step]
            ec = traj["exists_context"][self.current_step]
        else:
            vc = traj["vehicle_context"][-1]
            lc = traj["lane_context"][-1]
            ec = traj["exists_context"][-1]
        return vc, lc, ec

    def get_belief_state(self):
        """Concatenate history of states and contexts (vehicle/lane/exists) with left padding if needed."""
        s_hist = np.array(self.state_history)
        v_hist, l_hist, e_hist = [], [], []
        for v, l, e in self.context_history:
            v_hist.append(v)
            l_hist.append(l)
            e_hist.append(e)
        v_hist = np.array(v_hist)
        l_hist = np.array(l_hist)
        e_hist = np.array(e_hist)

        cur_len = len(self.state_history)
        if cur_len < self.history_length:
            pad_len = self.history_length - cur_len
            s_hist = np.vstack([np.zeros((pad_len, s_hist.shape[1])), s_hist])
            v_hist = np.vstack([np.zeros((pad_len, v_hist.shape[1])), v_hist])
            l_hist = np.vstack([np.zeros((pad_len, l_hist.shape[1])), l_hist])
            e_hist = np.vstack([np.zeros((pad_len, e_hist.shape[1])), e_hist])
        return np.hstack([s_hist, v_hist, l_hist, e_hist])
