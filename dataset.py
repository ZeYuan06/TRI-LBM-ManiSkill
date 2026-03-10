import os
import glob
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from concurrent.futures import ProcessPoolExecutor
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_axis_angle,
)


def debug_visualize_image(img_chw: np.ndarray, save_path: str = "debug_check.png"):
    import matplotlib.pyplot as plt

    img_hwc = np.transpose(img_chw, (1, 2, 0))
    plt.figure(figsize=(6, 6))
    plt.imshow(img_hwc)
    plt.title(f"Debug Frame")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[DEBUG] Saved visualization to {save_path}")


def load_single_episode(fpath):
    """
    Independent loading function, lightweight, only performs I/O and Numpy Reshape
    """
    with open(fpath, "rb") as f:
        ep_data = pickle.load(f)

    actions = ep_data["actions"]  # (L, A)
    base_rgb = ep_data["base_rgb"]  # uint8 (L, H, W, 3)
    wrist_rgb = ep_data["wrist_rgb"]  # uint8
    traj_state = ep_data["states"]

    obj_name = ep_data.get("object", "object")
    if isinstance(obj_name, str):
        parts = obj_name.split("_")
        if len(parts) > 1 and parts[0].isdigit():
            obj_name = " ".join(parts[1:])
        else:
            obj_name = obj_name.replace("_", " ")

    # NHWC -> NCHW (Numpy Transpose)
    base_rgb = np.transpose(base_rgb, (0, 3, 1, 2))
    wrist_rgb = np.transpose(wrist_rgb, (0, 3, 1, 2))

    obs_dict = {
        "base_rgb": base_rgb,  # uint8
        "wrist_rgb": wrist_rgb,  # uint8
        "state": traj_state,  # float64/32
    }

    return obs_dict, actions, {"instruction": f"pick up the {obj_name}"}


class RamenNormalizer:
    def __init__(
        self,
        data=None,
        norm_dims: list[int] = [0, 1, 2],
        gripper_dim: int = 9,
        state_dict=None,
    ):
        """
        Ramen Normalizer with special handling for Gripper.

        Args:
            norm_dims: List of dimensions for Ramen Normalization (Pos: 0, 1, 2)
            gripper_dim: Index for Gripper dimension (Simple MinMax)
        """
        if state_dict is not None:
            self.load_state_dict(state_dict)
            return

        if data is None:
            raise ValueError("Data must be provided for RamenNormalizer initialization")

        self.norm_dims = norm_dims
        self.gripper_dim = gripper_dim

        data = data.to("cpu")
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

        return

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

    def state_dict(self):
        """Export state as simple python types/numpy for safe checkpointing"""
        return {
            "norm_dims": self.norm_dims,
            "gripper_dim": self.gripper_dim,
            "x_02": self.x_02.cpu()
            .numpy()
            .tolist(),  # Save as list for max compatibility
            "x_98": self.x_98.cpu().numpy().tolist(),
            "gripper_min": float(self.gripper_min.cpu().item()),
            "gripper_max": float(self.gripper_max.cpu().item()),
        }

    def load_state_dict(self, state):
        """Restore state from dict"""
        self.norm_dims = state["norm_dims"]
        self.gripper_dim = state["gripper_dim"]

        # Restore Tensors
        self.x_02 = torch.tensor(state["x_02"])
        self.x_98 = torch.tensor(state["x_98"])
        self.gripper_min = torch.tensor(state["gripper_min"])
        self.gripper_max = torch.tensor(state["gripper_max"])

        # Aux params
        self.T = self.x_02.shape[0]
        self.D = self.x_02.shape[1]


class StandardNormalizer:
    def __init__(
        self,
        data=None,
        norm_dims: list[int] = [0, 1, 2, 3, 4, 5],
        gripper_dim: int = 6,
        state_dict=None,
    ):
        if state_dict is not None:
            self.load_state_dict(state_dict)
            return

        if data is None:
            raise ValueError("Data required for initialization")

        self.norm_dims = norm_dims
        self.gripper_dim = gripper_dim

        data_cpu = data.to("cpu").float()

        # Z-score Stats for Continuous Dims
        # data shape: (N, T, D) -> flatten to (N*T, D)
        flat_data = data_cpu.reshape(-1, data_cpu.shape[-1])

        self.mean = torch.mean(flat_data[:, norm_dims], dim=0)  # (len(norm_dims),)
        self.std = torch.std(flat_data[:, norm_dims], dim=0)  # (len(norm_dims),)

        self.std[self.std < 1e-6] = 1.0

        # MinMax Stats for Gripper
        gripper_data = flat_data[:, gripper_dim]
        self.gripper_min = torch.min(gripper_data)
        self.gripper_max = torch.max(gripper_data)

        if self.gripper_max - self.gripper_min < 1e-6:
            self.gripper_max += 1.0

        print(f"[StandardNormalizer] Init. Mean: {self.mean}, Std: {self.std}")

    def normalize(self, x):
        x_norm = x.clone()
        device = x.device

        # Z-score: (x - u) / std
        if len(self.norm_dims) > 0:
            target = x[..., self.norm_dims]
            mean = self.mean.to(device)
            std = self.std.to(device)
            x_norm[..., self.norm_dims] = (target - mean) / std

        # Gripper: Map to [-1, 1]
        if self.gripper_dim is not None:
            g_target = x[..., self.gripper_dim]
            g_min = self.gripper_min.to(device)
            g_max = self.gripper_max.to(device)
            x_norm[..., self.gripper_dim] = (
                2.0 * (g_target - g_min) / (g_max - g_min) - 1.0
            )

        return x_norm

    def unnormalize(self, x):
        x_unnorm = x.clone()
        device = x.device

        # Un-Z-score: x * std + u
        if len(self.norm_dims) > 0:
            target = x[..., self.norm_dims]
            mean = self.mean.to(device)
            std = self.std.to(device)
            x_unnorm[..., self.norm_dims] = target * std + mean

        # Un-Gripper
        if self.gripper_dim is not None:
            g_target = x[..., self.gripper_dim]
            g_min = self.gripper_min.to(device)
            g_max = self.gripper_max.to(device)
            x_unnorm[..., self.gripper_dim] = (g_target + 1.0) / 2.0 * (
                g_max - g_min
            ) + g_min

        return x_unnorm

    def state_dict(self):
        return {
            "norm_dims": self.norm_dims,
            "gripper_dim": self.gripper_dim,
            "mean": self.mean.cpu().numpy().tolist(),
            "std": self.std.cpu().numpy().tolist(),
            "gripper_min": float(self.gripper_min.cpu().item()),
            "gripper_max": float(self.gripper_max.cpu().item()),
        }

    def load_state_dict(self, state):
        self.norm_dims = state["norm_dims"]
        self.gripper_dim = state["gripper_dim"]
        self.mean = torch.tensor(state["mean"])
        self.std = torch.tensor(state["std"])
        self.gripper_min = torch.tensor(state["gripper_min"])
        self.gripper_max = torch.tensor(state["gripper_max"])


class MultiViewDataset(Dataset):
    def __init__(self, data_path, num_traj, stats_path=None):
        """
        paper config:
        obs_horizon = 2
        pred_horizon = 16
        """
        # --- Paper Constraints ---
        self.obs_horizon = 2
        self.pred_horizon = 16

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        print(
            f"Dataset Params Locked: Obs Horizon={self.obs_horizon}, Pred Horizon={self.pred_horizon}"
        )

        trajectories = {"observations": [], "actions": [], "meta": []}

        print(f"Loading pickle data from {data_path}...")
        file_pattern = os.path.join(data_path, "**", "trajectory.pkl")
        files = sorted(glob.glob(file_pattern, recursive=True))
        if num_traj is not None:
            files = files[:num_traj]

        print(f"Loading {len(files)} files in parallel...")
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(
                tqdm(executor.map(load_single_episode, files), total=len(files))
            )

        print("Converting to Tensor (Keeping uint8 for images)...")
        obs_traj_dict_list = []
        for obs_dict, actions, meta in results:
            obs_dict["base_rgb"] = torch.from_numpy(obs_dict["base_rgb"])
            obs_dict["wrist_rgb"] = torch.from_numpy(obs_dict["wrist_rgb"])
            obs_dict["state"] = torch.from_numpy(obs_dict["state"]).float()

            obs_traj_dict_list.append(obs_dict)
            trajectories["actions"].append(torch.from_numpy(actions).float())
            trajectories["meta"].append(meta)

        trajectories["observations"] = obs_traj_dict_list

        # Prepare Slicing & Padding Config
        self.slices = []
        pad_before = self.obs_horizon - 1
        num_traj = len(trajectories["actions"])
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            for start in range(-pad_before, L):
                self.slices.append((traj_idx, start))

        self.trajectories = trajectories

        if stats_path is not None and os.path.exists(stats_path):
            print(f"Loading pre-computed normalizer stats from {stats_path}...")
            state_dict = torch.load(stats_path)
            self.action_normalizer = RamenNormalizer(state_dict=state_dict["action"])
            self.state_normalizer = RamenNormalizer(state_dict=state_dict["state"])
            return

        # Ramen Normalization Preparation (Relativize Data for Stats)
        print("Calculating Delta Actions for Ramen Normalization...")

        # Sampling strategy to avoid OOM
        sample_size = min(10000, len(self.slices))
        sample_indices = np.random.choice(len(self.slices), sample_size, replace=False)

        delta_actions_cache = []
        states_cache = []

        for idx in tqdm(sample_indices, desc="Computing Norm Stats"):
            raw_sample = self._get_raw_sample(idx)

            # raw_sample: {'act_seq_delta': ..., 'state_abs': ...}
            delta_actions_cache.append(raw_sample["act_seq_delta"])
            states_cache.append(raw_sample["state_seq_abs"])

        all_delta_actions = torch.stack(delta_actions_cache)  # (N, Pred_T, D)
        all_states = torch.stack(states_cache)  # (N, Obs_T, D)

        # Initialize Normalizers:
        self.action_normalizer = RamenNormalizer(
            all_delta_actions,
            norm_dims=list(range(3)),  # Pos + Rotation
            gripper_dim=9,  # Gripper uses MinMax
        )

        # State: Same logic
        self.state_normalizer = RamenNormalizer(
            all_states, norm_dims=list(range(3)), gripper_dim=9
        )

        if stats_path is not None:
            print(f"Saving normalizer stats to {stats_path}...")
            torch.save(
                {
                    "action": self.action_normalizer.state_dict(),
                    "state": self.state_normalizer.state_dict(),
                },
                stats_path,
            )

        del delta_actions_cache, all_delta_actions, all_states

    def get_normalizers(self):
        return self.state_normalizer, self.action_normalizer

    def _get_raw_sample(self, index):
        traj_idx, start = self.slices[index]
        obs_traj = self.trajectories["observations"][traj_idx]
        act_traj = self.trajectories["actions"][traj_idx]
        L = act_traj.shape[0]

        # State Extraction
        obs_indices = np.arange(start, start + self.obs_horizon)
        clamped_obs_indices = np.clip(obs_indices, 0, L - 1)
        state_seq_abs = obs_traj["state"][clamped_obs_indices]  # (2, 10)

        # Action Extraction
        current_time_idx = start + self.obs_horizon - 1
        act_end_idx = current_time_idx + self.pred_horizon
        act_indices = np.arange(current_time_idx, act_end_idx)
        clamped_act_indices = np.clip(act_indices, 0, L - 1)

        act_seq_abs = act_traj[clamped_act_indices]  # (16, 10)
        curr_state_abs = state_seq_abs[-1]  # (10,)

        # Action Pos [0:3] - Current State Pos [0:3]
        act_pos_delta = act_seq_abs[:, :3] - curr_state_abs[:3]  # (16, 3)

        # Absolute Rotation 6D
        act_rot_6d_abs = act_seq_abs[:, 3:9]  # (16, 6)

        # Absolute Gripper
        act_gripper = act_seq_abs[:, 9:10]  # (16, 1)
        act_seq_delta = torch.cat([act_pos_delta, act_rot_6d_abs, act_gripper], dim=-1)

        return {
            "traj_idx": traj_idx,
            "obs_indices": clamped_obs_indices,  # For image loading
            "state_seq_abs": state_seq_abs,
            "act_seq_abs": act_seq_abs,
            "act_seq_delta": act_seq_delta,
            "instruction": self.trajectories["meta"][traj_idx]["instruction"],
        }

    def __getitem__(self, index):
        raw = self._get_raw_sample(index)

        traj_idx = raw["traj_idx"]
        obs_indices = raw["obs_indices"]
        obs_traj = self.trajectories["observations"][traj_idx]
        base_rgb = obs_traj["base_rgb"][obs_indices]
        wrist_rgb = obs_traj["wrist_rgb"][obs_indices]

        base_rgb = (base_rgb.float() / 255.0 - self.mean) / self.std
        wrist_rgb = (wrist_rgb.float() / 255.0 - self.mean) / self.std

        # Normalize State and Action
        norm_state = self.state_normalizer.normalize(
            raw["state_seq_abs"].unsqueeze(0)
        ).squeeze(0)

        norm_action = self.action_normalizer.normalize(
            raw["act_seq_delta"].unsqueeze(0)
        ).squeeze(0)

        # Construct Obs Dict
        obs_dict = {"base_rgb": base_rgb, "wrist_rgb": wrist_rgb, "state": norm_state}

        return {
            "observations": obs_dict,
            "actions": norm_action,
            "instruction": raw["instruction"],
        }

    def __len__(self):
        return len(self.slices)


if __name__ == "__main__":
    # --- Test Case ---
    print("\n" + "=" * 50)
    print("Running Dataset Integrity Test")
    print("=" * 50 + "\n")

    # Generate Dummy Data
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

        # Test __getitem__
        print("\nTesting item retrieval (idx=0)...")
        sample = dataset[0]

        obs = sample["observations"]
        actions = sample["actions"]
        instr = sample["instruction"]

        print(f"-> Instruction: '{instr}'")
        print(f"-> Actions Shape: {actions.shape} (Expected: 16, 10)")
        print(f"-> State Shape: {obs['state'].shape} (Expected: 2, 10)")
        print(f"-> RGB Shape: {obs['base_rgb'].shape} (Expected: 2, 3, 128, 128)")

        assert actions.shape == (16, 7), "Action shape mismatch!"
        assert obs["state"].shape == (2, 8), "State shape mismatch!"

        print("\nAll checks passed! Dataset logic is robust.")

    except Exception as e:
        print(f"\n!!! TEST FAILED with error: {e}")
        raise e
