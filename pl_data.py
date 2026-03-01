import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset import MultiViewDataset


class LBMDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=64, num_workers=8, num_traj=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_traj = num_traj

        self.stats_path = os.path.join(data_path, "dataset_stats.pt")

        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None

    def prepare_data(self):
        if not os.path.exists(self.stats_path):
            print(f"[Rank 0] Pre-calculating dataset stats to {self.stats_path}...")
            _ = MultiViewDataset(
                data_path=self.data_path,
                num_traj=self.num_traj,
                stats_path=self.stats_path,
            )

    def setup(self, stage=None):
        if self.train_dataset is None or self.val_dataset is None:
            self.full_dataset = MultiViewDataset(
                data_path=self.data_path, 
                num_traj=self.num_traj,
                stats_path=self.stats_path 
            )
            self.state_norm, self.action_norm = self.full_dataset.get_normalizers()

            total_len = len(self.full_dataset)
            train_len = int(0.9 * total_len)
            val_len = total_len - train_len
            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
            )
            print(f"[DataModule] Train: {train_len}, Val: {val_len}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
