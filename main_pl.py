import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch

from TRI_LBM.pl_model import LBMLightningModule
from pl_data import LBMDataModule

torch.set_float32_matmul_precision("high")


def get_args():
    parser = argparse.ArgumentParser(description="Train TRI-LBM with PyTorch Lightning")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--devices", type=str, default="1"
    )  # Can be int or comma-separated list of GPU IDs

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_traj", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_pl")
    parser.add_argument("--run_name", type=str, default="lbm_pl_run")
    parser.add_argument("--wandb_mode", type=str, default="online")

    parser.add_argument(
        "--eval_freq", type=int, default=5, help="Epoch frequency for validation"
    )
    parser.add_argument("--rollout_freq", type=int, default=15)

    return parser.parse_args()


def main():
    args = get_args()

    pl.seed_everything(42)

    dm = LBMDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_traj=args.num_traj,
    )
    dm.prepare_data()
    dm.setup()

    sample_item = dm.full_dataset[0]
    action_dim = sample_item["actions"].shape[-1]
    state_dim = sample_item["observations"]["state"].shape[-1]
    obs_horizon = sample_item["observations"]["base_rgb"].shape[0]

    print(
        f"Dims Check -> Action: {action_dim}, State: {state_dim}, Horizon: {obs_horizon}"
    )

    model = LBMLightningModule(
        action_dim=action_dim,
        dim_pose=state_dim,
        num_image_frames=obs_horizon,
        lr=args.lr,
        state_norm_params=dm.state_norm.state_dict(),
        action_norm_params=dm.action_norm.state_dict(),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="{epoch:02d}-{valloss:.4f}",
        save_top_k=2,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    epoch_checkpoint = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="{epoch:02d}",
        every_n_epochs=args.rollout_freq,
        save_top_k=-1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = WandbLogger(
        name=args.run_name,
        project="TRI-LBM-ManiSkill",
        entity="ai4ce",
        mode=args.wandb_mode,
    )

    devices = args.devices

    if "," in devices:
        devices = [int(x) for x in devices.split(",") if x.strip()]
        num_devices = len(devices)
    elif devices.isdigit():
        devices = int(devices)
        num_devices = devices

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if num_devices > 1 else "auto",
        precision="bf16-mixed",
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, epoch_checkpoint, lr_monitor],
        check_val_every_n_epoch=args.eval_freq,
        log_every_n_steps=10,
        sync_batchnorm=True,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
