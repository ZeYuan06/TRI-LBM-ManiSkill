import os
import glob
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def debug_visualize_image(img_hwc: np.ndarray, save_path: str = "debug_check.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(img_hwc)
    plt.title(f"Debug Frame")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[DEBUG] Saved visualization to {save_path}")


class RamenNormalizer:
    def __init__(self, data, norm_dims=[0, 1, 2], gripper_dim=9):
        """
        Ramen Normalizer with special handling for Gripper.

        Args:
            norm_dims: List of dimensions for Ramen Normalization (Pos: 0, 1, 2)
            gripper_dim: Index for Gripper dimension (Simple MinMax)
        """
        self.norm_dims = norm_dims
        self.gripper_dim = gripper_dim

        self.T = data.shape[1]
        self.D = data.shape[2]

        # --- A. Ramen Stats (Position) ---
        self.x_02 = torch.zeros((self.T, self.D))
        self.x_98 = torch.zeros((self.T, self.D))

        # --- B. MinMax Stats (Gripper) ---
        # Gripper usually doesn't need per-timestep stats if distribution is static (0 or 1),
        # but to keep consistency with structure, we can store scalar min/max.
        # However, simple min/max across WHOLE dataset is safer for binary data.
        gripper_data = data[..., gripper_dim]
        self.gripper_min = torch.min(gripper_data)
        self.gripper_max = torch.max(gripper_data)

        # Avoid division by zero if gripper never moves (constant value)
        if self.gripper_max - self.gripper_min < 1e-6:
            self.gripper_max += 1.0  # arbitrary scale, results in 0.0

        print(
            f"Computed Stats. Ramen Dims={norm_dims}, Gripper Dim={gripper_dim} (Range:[{self.gripper_min:.2f}, {self.gripper_max:.2f}])"
        )

        # Compute Ramen Stats for continuous dims
        for t in range(self.T):
            step_data = data[:, t, :].float()
            if len(norm_dims) > 0:
                q_vals = torch.quantile(
                    step_data[:, norm_dims], torch.tensor([0.02, 0.98]), dim=0
                )
                self.x_02[t, norm_dims] = q_vals[0]
                self.x_98[t, norm_dims] = q_vals[1]

    def normalize(self, x):
        """
        x: (..., T, D)
        """
        x_norm = x.clone()
        device = x.device

        # Ramen Normalization (Position)
        # y = 2 * (x - x02) / (x98 - x02) - 1
        if len(self.norm_dims) > 0:
            target = x[..., self.norm_dims]
            low = self.x_02[..., self.norm_dims].to(device)
            high = self.x_98[..., self.norm_dims].to(device)
            denom = high - low
            denom[denom < 1e-6] = 1.0

            val = 2.0 * (target - low) / denom - 1.0
            val = torch.clamp(val, -1.5, 1.5)
            x_norm[..., self.norm_dims] = val

        # MinMax Normalization (Gripper) -> Map to [-1, 1]
        # y = 2 * (x - min) / (max - min) - 1
        if self.gripper_dim is not None:
            g_target = x[..., self.gripper_dim]
            g_min = self.gripper_min.to(device)
            g_max = self.gripper_max.to(device)
            g_val = 2.0 * (g_target - g_min) / (g_max - g_min) - 1.0
            # No clustering necessary for binary, but good for safety
            g_val = torch.clamp(g_val, -1.0, 1.0)
            x_norm[..., self.gripper_dim] = g_val

        return x_norm

    def unnormalize(self, x):
        x_unnorm = x.clone()
        device = x.device

        # Un-Ramen (Position)
        if len(self.norm_dims) > 0:
            target = x[..., self.norm_dims]
            low = self.x_02[..., self.norm_dims].to(device)
            high = self.x_98[..., self.norm_dims].to(device)
            val = (target + 1.0) / 2.0 * (high - low) + low
            x_unnorm[..., self.norm_dims] = val

        # Un-MinMax (Gripper)
        if self.gripper_dim is not None:
            g_target = x[..., self.gripper_dim]
            g_min = self.gripper_min.to(device)
            g_max = self.gripper_max.to(device)
            g_val = (g_target + 1.0) / 2.0 * (g_max - g_min) + g_min
            x_unnorm[..., self.gripper_dim] = g_val

        return x_unnorm


class MultiViewDataset(Dataset):
    def __init__(self, data_path, num_traj):
        """
        paper config:
        obs_horizon = 2
        pred_horizon = 16
        """
        # --- Paper Constraints ---
        self.obs_horizon = 2
        self.pred_horizon = 16
        print(
            f"Dataset Params Locked: Obs Horizon={self.obs_horizon}, Pred Horizon={self.pred_horizon}"
        )

        trajectories = {"observations": [], "actions": [], "meta": []}

        print(f"Loading pickle data from {data_path}...")
        file_pattern = os.path.join(data_path, "**", "trajectory.pkl")
        files = sorted(glob.glob(file_pattern, recursive=True))
        if num_traj is not None:
            files = files[:num_traj]

        for fpath in tqdm(files, desc="Loading episodes"):
            with open(fpath, "rb") as f:
                ep_data = pickle.load(f)

            # Load Absolute Raw Data
            actions = ep_data["actions"]  # (L, A)
            base_rgb = ep_data["base_rgb"]
            wrist_rgb = ep_data["wrist_rgb"]
            traj_state = ep_data["states"]
            obj_name = ep_data.get("object", "object")
            if isinstance(obj_name, str):
                parts = obj_name.split("_")
                if len(parts) > 1 and parts[0].isdigit():
                    obj_name = " ".join(parts[1:])
                else:
                    obj_name = obj_name.replace("_", " ")

            # Padding L -> L+1
            base_rgb = np.concatenate([base_rgb, base_rgb[-1:]], axis=0)
            wrist_rgb = np.concatenate([wrist_rgb, wrist_rgb[-1:]], axis=0)
            traj_state = np.concatenate([traj_state, traj_state[-1:]], axis=0)

            # Transpose Images
            base_rgb = base_rgb.astype(np.float32) / 255.0
            wrist_rgb = wrist_rgb.astype(np.float32) / 255.0
            base_rgb = np.transpose(base_rgb, (0, 3, 1, 2))
            wrist_rgb = np.transpose(wrist_rgb, (0, 3, 1, 2))

            obs_dict = {
                "base_rgb": base_rgb,
                "wrist_rgb": wrist_rgb,
                "state": traj_state,  # Absolute Pose
            }
            meta_dict = {"instruction": f"pick up the {obj_name}"}

            trajectories["observations"].append(obs_dict)
            trajectories["actions"].append(actions)  # Absolute Pose
            trajectories["meta"].append(meta_dict)

        print("Data loaded. Converting to Tensor...")

        # Tensor Conversion
        obs_traj_dict_list = []
        for _obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict["base_rgb"] = torch.from_numpy(_obs_traj_dict["base_rgb"])
            _obs_traj_dict["wrist_rgb"] = torch.from_numpy(_obs_traj_dict["wrist_rgb"])
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"])
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list

        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i])

        # Prepare Slicing & Padding Config
        self.slices = []
        num_traj = len(trajectories["actions"])

        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            pad_before = self.obs_horizon - 1
            pad_after = self.pred_horizon - self.obs_horizon
            self.slices += [
                (traj_idx, start, start + self.pred_horizon)
                for start in range(-pad_before, L - self.pred_horizon + pad_after)
            ]

        self.trajectories = trajectories

        # -Ramen Normalization Preparation (Relativize Data for Stats)
        print("Pre-calculating Delta Actions for Ramen Normalization...")

        # Sampling strategy to avoid OOM
        sample_size = min(10000, len(self.slices))
        sample_indices = np.random.choice(len(self.slices), sample_size, replace=False)

        delta_actions_cache = []

        for idx in tqdm(sample_indices, desc="Computing Norm Stats"):
            traj_idx, start, end = self.slices[idx]
            L = self.trajectories["actions"][traj_idx].shape[0]

            # Get Absolute Action Sequence
            act_seq_abs = self.trajectories["actions"][traj_idx][
                max(0, start) : end
            ].float()

            # Pad logic (simplified just for stats)
            if start < 0:
                act_seq_abs = torch.cat(
                    [act_seq_abs[0].unsqueeze(0).repeat(-start, 1), act_seq_abs], dim=0
                )
            if end > L:
                # Replicate last (Absolute Pose Padding)
                last_frame = act_seq_abs[-1]
                act_seq_abs = torch.cat(
                    [act_seq_abs, last_frame.unsqueeze(0).repeat(end - L, 1)], dim=0
                )

            # Keep consistent length
            act_seq_abs = act_seq_abs[: self.pred_horizon]

            # Get Current Observation State (Reference Base)
            # Last frame of observation window
            curr_obs_idx = start + self.obs_horizon - 1
            if curr_obs_idx < 0:
                curr_obs_idx = 0
            if curr_obs_idx >= L:
                curr_obs_idx = L - 1
            curr_state_abs = self.trajectories["observations"][traj_idx]["state"][
                curr_obs_idx
            ].float()

            # Compute Delta (Action Relativization)
            # Pos (0-3) -> Relative; Rot (3-9) -> Absolute; Gripper (9) -> Absolute
            act_seq_delta = act_seq_abs.clone()
            act_seq_delta[:, :3] = act_seq_abs[:, :3] - curr_state_abs[:3]

            delta_actions_cache.append(act_seq_delta)

        all_delta_actions = torch.stack(delta_actions_cache)  # (N_samples, Pred_T, Dim)

        # States are normalized as Absolutes
        all_states_abs = torch.cat(
            [self.trajectories["observations"][i]["state"] for i in range(num_traj)],
            dim=0,
        ).unsqueeze(
            1
        )  # (Total_L, 1, D) for compat with Ramen

        # Initialize Normalizers:
        # Action:
        #   - Pos(0,1,2): Ramen
        #   - Rot(3-8): Skip (None)
        #   - Gripper(9): MinMax
        self.action_normalizer = RamenNormalizer(
            all_delta_actions,
            norm_dims=[0, 1, 2],  # Only Pos uses percentile
            gripper_dim=9,  # Gripper uses MinMax
        )

        # State: Same logic
        self.state_normalizer = RamenNormalizer(
            all_states_abs, norm_dims=[0, 1, 2], gripper_dim=9
        )

        del delta_actions_cache, all_delta_actions, all_states_abs

    def get_normalizers(self):
        return self.state_normalizer, self.action_normalizer

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        # --- 1. Load Observation ---
        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        # Slicing obs
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start) : start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        # --- 2. Load Action (Absolute) ---
        act_seq_abs = self.trajectories["actions"][traj_idx][max(0, start) : end]

        # Padding
        if start < 0:
            # FIX: Use unsqueeze for safety to match __init__ logic
            act_seq_abs = torch.cat(
                [act_seq_abs[0].unsqueeze(0).repeat(-start, 1), act_seq_abs], dim=0
            )
        if end > L:
            # FIX: Removed reference to undefined self.pad_action_arm
            # Now replicates the last frame (Absolute Pose padding)
            last_frame = act_seq_abs[-1]
            act_seq_abs = torch.cat(
                [act_seq_abs, last_frame.unsqueeze(0).repeat(end - L, 1)], dim=0
            )

        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq_abs.shape[0] == self.pred_horizon

        # --- 3. Relativization & Normalization ---
        # Get Reference Pose (Current Frame State)
        current_state_abs = obs_seq["state"][-1].clone()

        # Action -> Delta (Position Only)
        act_seq_delta = act_seq_abs.clone()
        act_seq_delta[:, :3] = act_seq_abs[:, :3] - current_state_abs[:3]

        # Normalize State (Abs -> Norm Abs)
        # Add dim for Ramen (T=1), remove after
        obs_seq["state"] = self.state_normalizer.normalize(
            obs_seq["state"].unsqueeze(0)
        ).squeeze(0)

        # Normalize Action (Delta -> Norm Delta)
        norm_action = self.action_normalizer.normalize(
            act_seq_delta.unsqueeze(0)
        ).squeeze(0)

        # --- 4. Get Instruction ---
        instruction = self.trajectories["meta"][traj_idx]["instruction"]

        return {
            "observations": obs_seq,
            "actions": norm_action,
            "instruction": instruction,
        }

    def __len__(self):
        return len(self.slices)


if __name__ == "__main__":
    # --- Test Case ---
    print("\n" + "=" * 50)
    print("Running Dataset Integrity Test")
    print("=" * 50 + "\n")

    # 1. Generate Dummy Data
    temp_dir = "./debug_data"
    print(f"Creating temp data at: {temp_dir}")

    try:
        print("\nInit Dataset...")
        dataset = MultiViewDataset(
            data_path=temp_dir,
            num_traj=None,
        )
        print("Dataset initialized successfully.")
        print(f"Total Slices: {len(dataset)}")

        # 3. Test __getitem__
        print("\nTesting item retrieval (idx=0)...")
        sample = dataset[0]

        obs = sample["observations"]
        actions = sample["actions"]
        instr = sample["instruction"]

        print(f"-> Instruction: '{instr}'")
        print(f"-> Actions Shape: {actions.shape} (Expected: 16, 10)")
        print(f"-> State Shape: {obs['state'].shape} (Expected: 2, 10)")
        print(f"-> RGB Shape: {obs['base_rgb'].shape} (Expected: 2, 3, 128, 128)")

        assert actions.shape == (16, 10), "Action shape mismatch!"
        assert obs["state"].shape == (2, 10), "State shape mismatch!"

        print("\nAll checks passed! Dataset logic is robust.")

    except Exception as e:
        print(f"\n!!! TEST FAILED with error: {e}")
        raise e
