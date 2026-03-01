import torch
import pytorch_lightning as pl
from torch import optim

from .lbm import LBM
from dataset import StandardNormalizer


class LBMLightningModule(pl.LightningModule):
    def __init__(
        self,
        action_dim,
        dim_pose,
        num_image_frames,
        lr=2e-4,
        state_norm_params=None,
        action_norm_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LBM(
            action_dim=action_dim,
            dim_pose=dim_pose,
            num_image_frames=num_image_frames,
        )

        self.state_norm = None
        self.action_norm = None

        if state_norm_params:
            self.state_norm = StandardNormalizer(state_dict=state_norm_params)

        if action_norm_params:
            self.action_norm = StandardNormalizer(state_dict=action_norm_params)

    def forward(self, text, images, pose, actions=None):
        return self.model(text, images, pose, actions)

    def training_step(self, batch, batch_idx):
        obs = batch["observations"]
        actions = batch["actions"]
        base_img = obs["base_rgb"]
        wrist_img = obs["wrist_rgb"]
        images = torch.stack([base_img, wrist_img], dim=1)  # (B, 2, T, C, H, W)
        state_seq = obs["state"]
        text_cmds = batch["instruction"]

        loss = self.model(
            text=text_cmds, images=images, pose=state_seq, actions=actions
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        obs = batch["observations"]
        actions = batch["actions"]
        base_img = obs["base_rgb"]
        wrist_img = obs["wrist_rgb"]
        images = torch.stack([base_img, wrist_img], dim=1)
        state_seq = obs["state"]
        text_cmds = batch["instruction"]

        loss = self.model(
            text=text_cmds, images=images, pose=state_seq, actions=actions
        )

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        vision_param_ids = set(map(id, self.model.image_model.parameters()))

        trainable_params_list = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        vision_params = [p for p in trainable_params_list if id(p) in vision_param_ids]
        base_params = [
            p for p in trainable_params_list if id(p) not in vision_param_ids
        ]

        optimizer = optim.AdamW(
            [
                {"params": base_params, "lr": self.hparams.lr},
                {"params": vision_params, "lr": self.hparams.lr * 0.1},
            ],
            lr=self.hparams.lr,
        )
        # Optional：Config LR Scheduler
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        # return [optimizer], [scheduler]
        return optimizer
