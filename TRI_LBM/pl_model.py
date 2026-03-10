import torch
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from copy import deepcopy


from .lbm import LBM
from dataset import RamenNormalizer


class EMACallback(Callback):
    def __init__(self, decay=0.999, update_every=1, update_after_step=0):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.ema_model = None

        self._ema_params = None
        self._model_params = None

    def on_fit_start(self, trainer, pl_module):
        if self.ema_model is None:
            self.ema_model = deepcopy(pl_module.model)
            print(f"[EMA] Initialized with decay {self.decay}")
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        self.ema_model.to(pl_module.device)

        ema_params, model_params = [], []
        for p_ema, p_model in zip(self.ema_model.parameters(), pl_module.model.parameters()):
            if p_ema.dtype.is_floating_point and p_model.dtype.is_floating_point:
                ema_params.append(p_ema)
                model_params.append(p_model)
        self._ema_params = ema_params
        self._model_params = model_params

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.ema_model is None:
            return
        step = trainer.global_step
        if step < self.update_after_step:
            return
        if (step % self.update_every) != 0:
            return

        with torch.no_grad():
            torch._foreach_mul_(self._ema_params, self.decay)
            torch._foreach_add_(self._ema_params, self._model_params, alpha=1.0 - self.decay)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.ema_model = deepcopy(pl_module.model)
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)
            print("[EMA] Loaded EMA state from checkpoint")


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
            self.state_norm = RamenNormalizer(state_dict=state_norm_params)

        if action_norm_params:
            self.action_norm = RamenNormalizer(state_dict=action_norm_params)

    def _get_ema_callback(self):
        if self.trainer is None:
            return None
        for cb in self.trainer.callbacks:
            if isinstance(cb, EMACallback):
                return cb
        return None

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

    def on_validation_epoch_start(self):
        self._ema_cb_ref = self._get_ema_callback()

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

        ema_model = self._ema_cb_ref.ema_model
        if ema_model is not None:
            ema_loss = ema_model(
                text=text_cmds, images=images, pose=state_seq, actions=actions
            )
            self.log(
                "val/ema_loss", ema_loss, on_step=False, on_epoch=True, sync_dist=True
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
