import argparse
import torch
import os

from TRI_LBM.pl_model import LBMLightningModule
from dataset import StandardNormalizer
from eval_utils import evaluate_rollout


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_test_args():
    parser = argparse.ArgumentParser(description="Test TRI-LBM Rollout")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_eval_episodes", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument(
        "--no_video", action="store_true", help="Disable video recording"
    )
    return parser.parse_args()


def main():
    args = get_test_args()
    device = torch.device(args.device)
    print(f"Testing checkpoint: {args.ckpt}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    print(f"Loading PL checkpoint: {args.ckpt}")

    pl_module = LBMLightningModule.load_from_checkpoint(args.ckpt, map_location=device)
    pl_module.eval()
    pl_module.freeze()

    model = pl_module.model.to(device)

    state_norm_params = pl_module.hparams.state_norm_params
    action_norm_params = pl_module.hparams.action_norm_params

    if state_norm_params is None or action_norm_params is None:
        raise ValueError(
            "Checkpoint does not contain normalizer params! Did you use the new pl_model?"
        )

    print("Restoring Normalizers from checkpoint...")
    state_norm = StandardNormalizer(state_dict=state_norm_params)
    action_norm = StandardNormalizer(state_dict=action_norm_params)

    obs_horizon = pl_module.hparams.num_image_frames

    evaluate_rollout(
        model=model,
        epoch="Test",
        device=device,
        state_norm=state_norm,
        action_norm=action_norm,
        num_eval_episodes=args.num_eval_episodes,
        obs_horizon=obs_horizon,
        max_steps=args.max_steps,
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
