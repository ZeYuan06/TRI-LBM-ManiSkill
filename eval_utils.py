import torch
import gymnasium as gym
from collections import deque
from pathlib import Path
from tqdm import tqdm
import wandb

from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_euler_angles,
    axis_angle_to_matrix,
)
from agent.widowx_env import PickBoxEnv


def debug_visualize_image(img_chw: torch.Tensor, save_path: str = "debug_check.png"):
    import matplotlib.pyplot as plt

    img_chw = img_chw.cpu().numpy()
    img_hwc = img_chw.transpose(1, 2, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_hwc)
    plt.title(f"Debug Frame")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[DEBUG] Saved visualization to {save_path}")


class DecoupledPose:
    """Minimal math wrapper to decouple coordinate-frame logic."""

    def __init__(self, pos: torch.Tensor, rot_mat: torch.Tensor):
        self.pos = pos
        self.rot_mat = rot_mat

    @classmethod
    def from_env_capture(cls, capture_tensor: torch.Tensor):
        """Parse env capture tensor [x, y, z, qx, qy, qz, qw, gripper]."""
        pos = capture_tensor[:, :3]
        quat = capture_tensor[:, 3:7]
        return cls(pos, quaternion_to_matrix(quat))

    def apply_delta(self, delta_pos: torch.Tensor, delta_axis_angle: torch.Tensor):
        """Compute target = anchor + predicted delta (pos + axis-angle rot)."""
        new_pos = self.pos + delta_pos
        rel_rot_mat = axis_angle_to_matrix(delta_axis_angle)
        new_rot_mat = torch.matmul(self.rot_mat, rel_rot_mat)
        return DecoupledPose(new_pos, new_rot_mat)

    def diff_to_env_action(
        self, target_pose: "DecoupledPose"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (world_position_delta, local_euler_delta).

        Position delta is in world frame. Rotation delta is expressed as
        local Euler angles: R_local = R_curr^T * R_target.
        """
        env_delta_pos = target_pose.pos - self.pos
        curr_rot_inv = self.rot_mat.transpose(-1, -2)
        env_delta_rot_mat = torch.matmul(curr_rot_inv, target_pose.rot_mat)
        env_delta_euler = matrix_to_euler_angles(env_delta_rot_mat, "XYZ")
        return env_delta_pos, env_delta_euler


def compute_step_delta_action(
    anchor: DecoupledPose,
    pred_delta_pos: torch.Tensor,
    pred_delta_axis_angle: torch.Tensor,
    pred_gripper: torch.Tensor,
    curr: DecoupledPose,
    pos_limit: float = 0.05,
    rot_limit: float = 0.15,
):
    """Compute a safe, clipped corrective action that moves current -> predicted target.

    Steps:
      1) Build absolute target pose = anchor + predicted delta.
      2) Compute corrective delta from current to target (world position, local Euler).
      3) Clip position and rotation deltas to provided safety limits and return action.
    """
    # Absolute target pose from anchor + predicted delta
    target = anchor.apply_delta(pred_delta_pos, pred_delta_axis_angle)

    # Corrective delta to apply in the environment frame
    env_delta_pos, env_delta_euler = curr.diff_to_env_action(target)

    # Clip position delta elementwise to [-pos_limit, pos_limit]
    clipped_pos_delta = torch.clamp(env_delta_pos, -pos_limit, pos_limit)

    # Clip rotation delta by limiting its magnitude per sample to rot_limit
    rot_norm = torch.linalg.norm(env_delta_euler, dim=1, keepdim=True)
    scale_factor = torch.ones_like(rot_norm)
    mask = rot_norm > rot_limit
    if mask.any():
        scale_factor[mask] = rot_limit / (rot_norm[mask] + 1e-8)
    clipped_euler_delta = env_delta_euler * scale_factor

    # Return concatenated action: [dx,dy,dz, droll,dpitch,dyaw, gripper]
    return torch.cat([clipped_pos_delta, clipped_euler_delta, pred_gripper], dim=1)


class StateCaptureWrapper(gym.Wrapper):
    def capture_state(self):
        """Capture current robot state (TCP pose, gripper) directly from agent. Must be called at the correct time!"""
        env = self.env.unwrapped

        # Get end-effector (TCP) pose (Batch, 7)
        tcp_pose = env.agent.tcp_pose
        tcp_pos = tcp_pose.p
        tcp_quat = tcp_pose.q

        # rotation_matrix = quaternion_to_matrix(tcp_quat)

        # # 6D rotation representation: first two rows of the rotation matrix
        # # (Batch, 3, 3) -> (Batch, 2, 3) -> (Batch, 6)
        # rot_6d = rotation_matrix[:, :2, :].reshape(-1, 6)

        # Gripper position (usually last dim of qpos)
        gripper_position = env.agent.robot.get_qpos()[:, -1:]

        # Return concatenated: [xyz(3), rot6d(6), gripper(1)] -> (Batch, 10)
        return torch.cat([tcp_pos, tcp_quat, gripper_position], dim=1)


def evaluate_rollout(
    model,
    epoch,
    device,
    state_norm,
    action_norm,
    num_eval_episodes: int,
    obs_horizon=2,
    max_steps=200,
    save_video=True,
):
    """
    Perform environment rollout to evaluate success rate.
    
    Args:
        model: The model to evaluate.
        epoch: Current training epoch.
        device: Device to run the evaluation on (e.g., 'cuda').
        state_norm: Normalization utility for state inputs.
        action_norm: Normalization utility for action outputs.
        num_eval_episodes: Number of evaluation episodes to run.
        obs_horizon: Number of historical observations to consider.
        max_steps: Maximum steps per episode.
        save_video: Whether to save rollout videos.

    Returns:
        avg_metrics: Dictionary containing average metrics such as success rate and return.
    """
    print(f"\n[Epoch {epoch}] Starting Rollout Evaluation...")
    model.eval()

    assets_dir = Path(__file__).parent / "YCB_processed"
    all_obj_paths = []
    box_obj_path = None
    if assets_dir.exists():
        for item in assets_dir.iterdir():
            if item.is_dir():
                obj_file = item / "textured.obj"
                if obj_file.exists():
                    all_obj_paths.append(str(obj_file))

        potential_box = assets_dir / "005_tomato_soup_can" / "textured.obj"
        if potential_box.exists():
            box_obj_path = str(potential_box)
    else:
        raise FileNotFoundError(f"Assets directory {assets_dir} does not exist.")

    num_envs = 32
    env = ManiSkillVectorEnv(
        env=PickBoxEnv.uid,
        num_envs=num_envs,
        auto_reset=True,
        record_metrics=True,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        box_obj_path=box_obj_path,
        distractor_pool_paths=all_obj_paths if all_obj_paths else None,
        num_distractors=0,
        max_episode_steps=max_steps,
    )
    env = StateCaptureWrapper(env)

    if save_video:
        env = RecordEpisode(
            env,
            output_dir=f"./rollout_videos/Epoch{epoch}",
            max_steps_per_video=max_steps * 4,
        )

    instruction = "pick up the tomato soup can"

    # Paper Configuration
    EXECUTION_HORIZON = 8  # Re-plan after executing H steps

    # History Buffers
    # Use deque to store the most recent obs_horizon frames
    pose_history = deque(maxlen=obs_horizon)
    base_rgb_history = deque(maxlen=obs_horizon)
    wrist_rgb_history = deque(maxlen=obs_horizon)

    # Metrics accumulator
    eps_count = 0
    pbar = tqdm(total=num_eval_episodes, desc="Evaluating")

    metrics_accumulator = {
        "success_once": 0.0,
        "success_at_end": 0.0,
        "return": 0.0,
    }

    # CLIP normalization constants
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(
        1, 3, 1, 1
    )
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(
        1, 3, 1, 1
    )

    def get_formatted_images(obs_dict):
        """
        Format and normalize images from the observation dictionary.
        Args:
            obs_dict: Dictionary containing sensor data.
        Returns:
            base: Normalized base camera image.
            wrist: Normalized wrist camera image.
        """
        base = obs_dict["sensor_data"]["base_camera"]["rgb"].float() / 255.0
        wrist = obs_dict["sensor_data"]["wrist_camera"]["rgb"].float() / 255.0

        base = base.permute(0, 3, 1, 2)
        wrist = wrist.permute(0, 3, 1, 2)
        base = (base - CLIP_MEAN) / CLIP_STD
        wrist = (wrist - CLIP_MEAN) / CLIP_STD

        return base, wrist

    with torch.no_grad():
        # Reset the environment and initialize history buffers
        obs, _ = env.reset()
        init_base, init_wrist = get_formatted_images(obs)
        init_pose = env.capture_state().to(device)
        for _ in range(obs_horizon):
            base_rgb_history.append(init_base)
            wrist_rgb_history.append(init_wrist)
            pose_history.append(init_pose)

        while eps_count < num_eval_episodes:
            # Re-planning
            base_seq = torch.stack(list(base_rgb_history), dim=1).to(device)
            wrist_seq = torch.stack(list(wrist_rgb_history), dim=1).to(device)
            img_input = torch.stack([base_seq, wrist_seq], dim=1)

            pose_seq = torch.stack(list(pose_history), dim=1).to(device)
            pose_input = state_norm.normalize(pose_seq)

            # Predict future trajectory
            pred_actions_norm = model.sample(
                text=[instruction] * num_envs, images=img_input, pose=pose_input
            )  # (B, 16, 10)

            pred_actions_raw = action_norm.unnormalize(pred_actions_norm)
            horizon = min(EXECUTION_HORIZON, pred_actions_raw.shape[1])
            anchor_state = env.capture_state().to(device)
            anchor = DecoupledPose.from_env_capture(anchor_state)

            # Execution Loop
            for t in range(horizon):
                step_action = pred_actions_raw[:, t, :]
                pred_delta_pos = step_action[:, 0:3]
                pred_delta_aa = step_action[:, 3:6]
                pred_gripper = step_action[:, 6:7]

                curr_state = env.capture_state().to(device)
                curr = DecoupledPose.from_env_capture(curr_state)

                env_action = compute_step_delta_action(
                    anchor=anchor,
                    pred_delta_pos=pred_delta_pos,
                    pred_delta_axis_angle=pred_delta_aa,
                    pred_gripper=pred_gripper,
                    curr=curr,
                    pos_limit=0.05,
                    rot_limit=0.15,
                )

                obs, reward, terminated, truncated, info = env.step(env_action)
                curr_base, curr_wrist = get_formatted_images(obs)
                curr_pose = env.capture_state().to(device)
                base_rgb_history.append(curr_base)
                wrist_rgb_history.append(curr_wrist)
                pose_history.append(curr_pose)

                dones = terminated | truncated
                if dones.any():
                    num_done_now = dones.sum().item()
                    final_info = info["final_info"]
                    success_once = final_info["episode"]["success_once"]
                    metrics_accumulator["success_once"] += (
                        (success_once & dones).sum().item()
                    )
                    success = final_info["success"]  # (B,) bool
                    metrics_accumulator["success_at_end"] += (
                        (success & dones).sum().item()
                    )
                    epi_return = final_info["episode"]["return"]
                    metrics_accumulator["return"] += epi_return[dones].sum().item()

                    eps_count += num_done_now
                    pbar.update(num_done_now)

                    pose_history.clear()
                    base_rgb_history.clear()
                    wrist_rgb_history.clear()
                    for _ in range(obs_horizon):
                        pose_history.append(curr_pose)
                        base_rgb_history.append(curr_base)
                        wrist_rgb_history.append(curr_wrist)

                    break
    pbar.close()
    env.close()

    avg_metrics = {}
    if eps_count > 0:
        avg_metrics["success_rate/once"] = (
            metrics_accumulator["success_once"] / eps_count
        )
        avg_metrics["success_rate/at_end"] = (
            metrics_accumulator["success_at_end"] / eps_count
        )
        avg_metrics["return"] = metrics_accumulator["return"] / eps_count
    else:
        for k in metrics_accumulator:
            avg_metrics[k] = 0.0
    print(f"Rollout Result ({eps_count} eps):")
    print(f"  Success Rate (Once):   {avg_metrics['success_rate/once']*100:.1f}%")
    print(f"  Success Rate (At End): {avg_metrics['success_rate/at_end']*100:.1f}%")
    print(f"  Avg Return:            {avg_metrics['return']:.4f}")

    if wandb.run is not None:
        wandb_log_dict = {f"rollout/{k}": v for k, v in avg_metrics.items()}
        wandb_log_dict["epoch"] = epoch
        wandb.log(wandb_log_dict)

    return avg_metrics
