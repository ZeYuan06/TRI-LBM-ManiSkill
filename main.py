import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import gymnasium as gym
from collections import deque
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_euler_angles,
)

from TRI_LBM import LBM
from dataset import MultiViewDataset
from agent.widowx_env import PickBoxEnv


def get_args():
    parser = argparse.ArgumentParser(description="Train TRI-LBM")

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Path to trajectory pickles"
    )
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--num_traj",
        type=int,
        default=None,
        help="Limit number of trajectories for debugging",
    )

    parser.add_argument(
        "--eval_freq", type=int, default=5, help="Epoch frequency for validation"
    )
    parser.add_argument(
        "--rollout_freq", type=int, default=10, help="Epoch frequency for env rollout"
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=20,
        help="Total number of episodes to evaluate during rollout",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode during rollout evaluation",
    )

    args = parser.parse_args()
    return args


def evaluate_loss(model, dataloader, device):
    """在验证集上计算 Loss"""
    model.eval()
    total_loss = 0
    count = 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            obs = batch["observations"]
            actions = batch["actions"].to(device)  # (B, T_pred, Act_Dim)

            # State: (B, T_obs, 10)
            state_seq = obs["state"].to(device, non_blocking=True)
            state = state_seq.flatten(start_dim=1)  # (B, 20)

            # Images: (B, T, C, H, W)
            base_img = obs["base_rgb"].to(device)
            wrist_img = obs["wrist_rgb"].to(device)

            # Paper/Code Logic: (Base T0, Base T1, Wrist T0, Wrist T1)
            images = torch.cat([base_img, wrist_img], dim=1)  # (B, 2*T, C, H, W)
            images = images.permute(0, 2, 1, 3, 4)

            # Text: List[str]
            text_cmds = batch["instruction"]

            loss = model(text=text_cmds, images=images, pose=state, actions=actions)

            total_loss += loss.item() * actions.shape[0]
            count += actions.shape[0]
            pbar.set_postfix({"val_loss": total_loss / count})

    return total_loss / count


class StateCaptureWrapper(gym.Wrapper):
    def capture_state(self):
        """Capture current robot state (TCP pose, gripper) directly from agent. Must be called at the correct time!"""
        env = self.env.unwrapped

        # Get end-effector (TCP) pose (Batch, 7)
        tcp_pose = env.agent.tcp_pose
        tcp_pos = tcp_pose.p
        tcp_quat = tcp_pose.q

        rotation_matrix = quaternion_to_matrix(tcp_quat)

        # 6D rotation representation: first two rows of the rotation matrix
        # (Batch, 3, 3) -> (Batch, 2, 3) -> (Batch, 6)
        rot_6d = rotation_matrix[:, :2, :].reshape(-1, 6)

        # Gripper position (usually last dim of qpos)
        gripper_position = env.agent.robot.get_qpos()[:, -1:]

        # Return concatenated: [xyz(3), rot6d(6), gripper(1)] -> (Batch, 10)
        return torch.cat([tcp_pos, rot_6d, gripper_position], dim=1)


def evaluate_rollout(
    model,
    epoch,
    device,
    state_norm,
    action_norm,
    num_eval_episodes,
    obs_horizon=2,
    max_steps=200,
):
    """
    执行环境 Rollout 测试成功率
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

    num_envs = 2
    env = ManiSkillVectorEnv(
        env=PickBoxEnv.uid,
        num_envs=num_envs,
        auto_reset=True,
        record_metrics=True,
        obs_mode="rgb",
        control_mode="pd_ee_pose",
        render_mode="rgb_array",
        box_obj_path=box_obj_path,
        distractor_pool_paths=all_obj_paths if all_obj_paths else None,
        num_distractors=0,
        max_episode_steps=max_steps,
    )
    env = StateCaptureWrapper(env)
    env = RecordEpisode(
        env, output_dir="./rollout_videos", max_steps_per_video=max_steps * 4
    )

    instruction = "pick up the tomato soup can"

    # Paper Config
    EXECUTION_HORIZON = 8  # 执行 H 步后重新规划

    # --- 3. 历史 buffers ---
    # 使用 deque 存储最近 obs_horizon 帧
    pose_history = deque(maxlen=obs_horizon)
    base_rgb_history = deque(maxlen=obs_horizon)
    wrist_rgb_history = deque(maxlen=obs_horizon)

    # 统计数据
    success_count = 0
    eps_count = 0
    pbar = tqdm(total=num_eval_episodes, desc="Evaluating")

    # 辅助函数：格式化图像
    def get_formatted_images(obs_dict):
        base = obs_dict["sensor_data"]["base_camera"]["rgb"].float() / 255.0
        wrist = obs_dict["sensor_data"]["wrist_camera"]["rgb"].float() / 255.0
        return base.permute(0, 3, 1, 2), wrist.permute(0, 3, 1, 2)

    with torch.no_grad():
        # Reset 并初始化历史
        obs, _ = env.reset()
        init_base, init_wrist = get_formatted_images(obs)
        init_pose = env.capture_state().to(device)
        for _ in range(obs_horizon):
            base_rgb_history.append(init_base)
            wrist_rgb_history.append(init_wrist)
            pose_history.append(init_pose)

        # --- 主循环: 模仿 evaluate.py 的结构 ---
        while eps_count < num_eval_episodes:
            # Re-planning
            base_seq = torch.stack(list(base_rgb_history), dim=1).to(device)
            wrist_seq = torch.stack(list(wrist_rgb_history), dim=1).to(device)
            img_input = torch.cat([base_seq, wrist_seq], dim=1).permute(
                0, 2, 1, 3, 4
            )  # (B, C, 2T, H, W)

            pose_seq = torch.stack(list(pose_history), dim=1).to(device)
            pose_seq_norm = state_norm.normalize(pose_seq)
            pose_input = pose_seq_norm.flatten(start_dim=1)

            # Predict future trajectory
            pred_actions_norm = model.sample(
                text=[instruction] * num_envs, images=img_input, pose=pose_input
            )  # (B, 16, 10)

            pred_actions_raw = action_norm.unnormalize(pred_actions_norm)
            horizon = min(EXECUTION_HORIZON, pred_actions_raw.shape[1])
            planning_base_pos = env.capture_state().to(device)[:, :3].clone()

            # Execution Loop
            for t in range(horizon):
                # World Frame Delta
                raw_action = pred_actions_raw[:, t, :]
                target_action_world = raw_action.clone()
                target_action_world[:, :3] = (
                    raw_action[:, :3] + planning_base_pos
                )  # Apply Delta

                # World -> Local (Batch Operation)
                target_pos_world = target_action_world[:, :3]
                target_rot6d_world = target_action_world[:, 3:9]
                target_gripper = target_action_world[:, 9:]

                r1 = target_rot6d_world[:, :3]
                r2 = target_rot6d_world[:, 3:]
                x = r1 / torch.norm(r1, dim=1, keepdim=True)
                z = torch.cross(x, r2)
                z = z / torch.norm(z, dim=1, keepdim=True)
                y = torch.cross(z, x)
                target_rot_mat_world = torch.stack([x, y, z], dim=2)

                home_pose = Pose.create(
                    env.unwrapped.agent.keyframes["home"].pose,
                    device=env.get_wrapper_attr("device"),
                )
                root_pos = home_pose.p
                root_quat = home_pose.q
                root_rot_mat = quaternion_to_matrix(root_quat)

                pos_local = torch.matmul(
                    root_rot_mat.transpose(1, 2),
                    (target_pos_world - root_pos).unsqueeze(-1),
                ).squeeze(-1)
                rot_mat_local = torch.matmul(
                    root_rot_mat.transpose(1, 2), target_rot_mat_world
                )
                euler_local = matrix_to_euler_angles(rot_mat_local, "XYZ")

                env_action = torch.cat([pos_local, euler_local, target_gripper], dim=1)

                obs, reward, terminated, truncated, info = env.step(env_action)
                curr_base, curr_wrist = get_formatted_images(obs)
                curr_pose = env.capture_state().to(device)
                base_rgb_history.append(curr_base)
                wrist_rgb_history.append(curr_wrist)
                pose_history.append(curr_pose)

                dones = terminated | truncated
                if dones.any():
                    final_info = info["final_info"]
                    batch_success = final_info["episode"]["success_once"]
                    num_done_now = dones.sum().item()
                    num_succ_now = (batch_success & dones).sum().item()

                    success_count += num_succ_now
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

    # 汇总结果
    success_rate = success_count / eps_count if eps_count > 0 else 0
    print(f"Rollout Result: {success_rate*100:.1f}% ({success_count}/{eps_count})")


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running on {device}")

    # 1. 加载数据集 (移除 args 参数)
    full_dataset = MultiViewDataset(
        data_path=args.data_path,
        num_traj=args.num_traj,
    )

    # 获取归一化器
    state_norm, action_norm = full_dataset.get_normalizers()

    # 2. 划分数据集
    total_len = len(full_dataset)
    train_len = int(0.9 * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 3. 初始化模型
    sample = full_dataset[0]
    action_dim = sample["actions"].shape[-1]
    state_dim = sample["observations"]["state"].shape[-1]
    obs_horizon = sample["observations"]["base_rgb"].shape[0]
    num_cameras = 2  # Base + Wrist
    total_frames = obs_horizon * num_cameras
    flat_pose_dim = state_dim * obs_horizon
    print(
        f"Dims -> Action: {action_dim}, State: {state_dim}, Img Frames: {total_frames}"
    )

    model = LBM(
        action_dim=action_dim,
        dim_pose=flat_pose_dim,
        num_image_frames=total_frames,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. 训练循环
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            # 从 batch 获取数据 (已经在 GPU)
            # Obs: {"base_rgb": (B,T,C,H,W), "state": (B,T,10)}
            obs = batch["observations"]
            actions = batch["actions"].to(device)  # (B, 16, 10)

            # 获取图片 (B, T, C, H, W) Float [0,1]
            base_img = obs["base_rgb"].to(device)
            wrist_img = obs["wrist_rgb"].to(device)
            images = torch.cat([base_img, wrist_img], dim=1)
            images = images.permute(0, 2, 1, 3, 4)

            # 获取当前帧 Pose (Normalized)
            state_seq = obs["state"].to(device)  # (B, 2, 10)
            state = state_seq.flatten(start_dim=1)  # (B, 20)

            # 获取文本 (List of str)
            text_cmds = batch["instruction"]  # List[str]

            # Forward
            loss = model(text=text_cmds, images=images, pose=state, actions=actions)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Avg Loss: {avg_train_loss:.4f}")

        # 验证
        if epoch % args.eval_freq == 0:
            val_loss = evaluate_loss(model, val_loader, device)
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pt")
                )
                print("Saved Best Model")

        # Rollout
        if epoch % args.rollout_freq == 0:
            evaluate_rollout(
                model,
                epoch,
                device,
                state_norm,
                action_norm,
                num_eval_episodes=args.num_eval_episodes,
                obs_horizon=obs_horizon,
                max_steps=args.max_steps,
            )

        torch.save(model.state_dict(), os.path.join(args.save_dir, "latest.pt"))


if __name__ == "__main__":
    main()
