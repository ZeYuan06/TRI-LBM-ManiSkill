import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from TRI_LBM import LBM
from dataset import MultiViewDataset
from eval_utils import evaluate_rollout
from agent.widowx_env import PickBoxEnv


def get_args():
    parser = argparse.ArgumentParser(description="Train TRI-LBM")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=150)
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
        "--rollout_freq", type=int, default=15, help="Epoch frequency for env rollout"
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=32,
        help="Total number of episodes to evaluate during rollout",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode during rollout evaluation",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="WandB logging mode",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="debug_run",
        help="Name for this training run (used in WandB)",
    )

    args = parser.parse_args()
    return args


def evaluate_loss(model, dataloader, device):
    """Evaluate the model on the validation set and return average loss."""
    model.eval()
    total_loss = 0
    count = 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            obs = batch["observations"]
            actions = batch["actions"].to(device)  # (B, T_pred, Act_Dim)

            # State: (B, T_obs, 10)
            state = obs["state"].to(device)

            # Images: (B, T, C, H, W)
            base_img = obs["base_rgb"].to(device)
            wrist_img = obs["wrist_rgb"].to(device)

            # Paper/Code Logic: (Base T0, Base T1, Wrist T0, Wrist T1)
            images = torch.stack([base_img, wrist_img], dim=1)  # (B, 2, T, C, H, W)

            # Text: List[str]
            text_cmds = batch["instruction"]

            loss = model(text=text_cmds, images=images, pose=state, actions=actions)

            total_loss += loss.item() * actions.shape[0]
            count += actions.shape[0]
            pbar.set_postfix({"val_loss": total_loss / count})

    avg_val_loss = total_loss / count
    wandb.log({"val/loss": avg_val_loss})

    return avg_val_loss


def main():
    args = get_args()

    wandb.init(
        name=args.run_name,
        entity="ai4ce",
        project="TRI-LBM-ManiSkill",
        config=vars(args),
        mode=args.wandb_mode,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running on {device}")

    full_dataset = MultiViewDataset(
        data_path=args.data_path,
        num_traj=args.num_traj,
    )
    state_norm, action_norm = full_dataset.get_normalizers()

    total_len = len(full_dataset)
    train_len = int(0.9 * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    sample = full_dataset[0]
    action_dim = sample["actions"].shape[-1]
    state_dim = sample["observations"]["state"].shape[-1]
    obs_horizon = sample["observations"]["base_rgb"].shape[0]
    print(
        f"Dims -> Action: {action_dim}, State: {state_dim}, Obs Horizon: {obs_horizon}"
    )

    model = LBM(
        action_dim=action_dim,
        dim_pose=state_dim,
        num_image_frames=obs_horizon,
    ).to(device)

    total_params = 0
    trainable_params = 0
    frozen_params = 0
    vision_params_count = 0
    vision_param_ids = set(map(id, model.image_model.parameters()))

    for p in model.parameters():
        num_params = p.numel()
        total_params += num_params

        if p.requires_grad:
            trainable_params += num_params
            if id(p) in vision_param_ids:
                vision_params_count += num_params
        else:
            frozen_params += num_params

    base_trainable_count = trainable_params - vision_params_count

    print(
        f"Total Params: {total_params:,} | Frozen Params: {frozen_params:,} | Trainable Params: {trainable_params:,}"
    )
    print(
        f"Base (1.0x LR): {base_trainable_count:,} | Vision (0.1x LR):{vision_params_count:,}"
    )

    trainable_params_list = list(filter(lambda p: p.requires_grad, model.parameters()))
    vision_params = [p for p in trainable_params_list if id(p) in vision_param_ids]
    base_params = [p for p in trainable_params_list if id(p) not in vision_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": vision_params, "lr": args.lr * 0.1},
        ],
        lr=args.lr,
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            optimizer.zero_grad()
            # Obs: {"base_rgb": (B,T,C,H,W), "state": (B,T,10)}
            obs = batch["observations"]
            actions = batch["actions"].to(device)  # (B, 16, 10)

            base_img = obs["base_rgb"].to(device)  # (B, T, C, H, W)
            wrist_img = obs["wrist_rgb"].to(device)  # (B, T, C, H, W)
            images = torch.stack([base_img, wrist_img], dim=1)

            state = obs["state"].to(device)  # (B, 2, 10)
            text_cmds = batch["instruction"]  # List[str]

            # Forward
            loss = model(text=text_cmds, images=images, pose=state, actions=actions)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({"train/step_loss": loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Avg Loss: {avg_train_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})

        # Validation
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

    wandb.finish()


if __name__ == "__main__":
    main()
