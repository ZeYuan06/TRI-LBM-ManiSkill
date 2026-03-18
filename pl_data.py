import os
import glob
import random
import collections
from tqdm import tqdm
import pickle
import torch
import pytorch_lightning as pl
import open_clip
from torch.utils.data import DataLoader
from dataset import MultiViewDataset, RamenNormalizer


class LBMDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=64, num_workers=8, num_traj=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_traj = num_traj

        self.cache_path = os.path.join(data_path, f"dataset_cache_{num_traj}.pt")

    def prepare_data(self):
        if os.path.exists(self.cache_path):
            print(f"[Rank 0] Cache exists at {self.cache_path}, skipping preparation.")
            return

        print(f"[Rank 0] Generating dataset cache: {self.cache_path}...")

        file_pattern = os.path.join(self.data_path, "**", "trajectory.pkl")
        all_files = sorted(glob.glob(file_pattern, recursive=True))

        if self.num_traj is not None and self.num_traj < len(all_files):
            print(
                f"Stratified subsampling {self.num_traj} trajectories based on obj_name..."
            )

            obj_to_files = collections.defaultdict(list)
            for fpath in tqdm(all_files, desc="Scanning metadata"):
                with open(fpath, "rb") as f:
                    ep_data = pickle.load(f)
                    obj_name = ep_data.get("object", "unknown")
                    if isinstance(obj_name, str):
                        parts = obj_name.split("_")
                        if len(parts) > 1 and parts[0].isdigit():
                            obj_name = " ".join(parts[1:])
                        else:
                            obj_name = obj_name.replace("_", " ")

                    obj_to_files[obj_name].append(fpath)

            sampled_files = []
            num_categories = len(obj_to_files)
            print(
                f"Found {num_categories} unique objects. Distributing {self.num_traj} samples..."
            )

            target_per_obj = self.num_traj // num_categories
            remainder = self.num_traj % num_categories

            for obj_name, paths in obj_to_files.items():
                random.shuffle(paths)

                quota = target_per_obj + (1 if remainder > 0 else 0)
                if remainder > 0:
                    remainder -= 1

                take = min(quota, len(paths))
                sampled_files.extend(paths[:take])

            if len(sampled_files) < self.num_traj:
                remaining_files = list(set(all_files) - set(sampled_files))
                random.shuffle(remaining_files)
                sampled_files.extend(
                    remaining_files[: self.num_traj - len(sampled_files)]
                )

            print(
                f"Subsampling complete. Final list contains {len(sampled_files)} files."
            )
        else:
            print(f"Using all {len(all_files)} trajectories without subsampling.")
            sampled_files = all_files

        random.shuffle(sampled_files)
        train_len = int(0.9 * len(sampled_files))
        train_files = sampled_files[:train_len]
        val_files = sampled_files[train_len:]

        print("[Rank 0] Calculating Normalization Stats from Training set...")
        temp_train_dataset = MultiViewDataset(
            file_list=train_files, state_norm=None, action_norm=None
        )
        sample_size = min(10000, len(temp_train_dataset))
        sample_indices = random.sample(range(len(temp_train_dataset)), sample_size)
        delta_actions_cache = []
        states_cache = []

        for idx in tqdm(sample_indices, desc="Computing Norm Stats"):
            raw_sample = temp_train_dataset._get_raw_sample(idx)
            delta_actions_cache.append(raw_sample["act_seq_delta"])
            states_cache.append(raw_sample["state_seq_abs"])

        all_delta_actions = torch.stack(delta_actions_cache)  # (N, Pred_T, D)
        all_states = torch.stack(states_cache)  # (N, Obs_T, D)

        action_normalizer = RamenNormalizer(
            all_delta_actions,
            norm_dims=list(range(3)),  # Pos dims
            gripper_dim=9,  # Gripper dims
        )

        state_normalizer = RamenNormalizer(
            all_states, norm_dims=list(range(3)), gripper_dim=9
        )

        print("[Rank 0] Computing Text Embedding Cache...")
        temp_clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        temp_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        temp_clip_model = temp_clip_model.to(device).eval()
        
        text_embedding_dict = {}
        unique_instructions = set(raw_sample["instruction"] for raw_sample in temp_train_dataset.trajectories["meta"])
        
        with torch.no_grad():
            for inst in unique_instructions:
                tokens = temp_tokenizer([inst]).to(device)
                emb = temp_clip_model.encode_text(tokens)
                text_embedding_dict[inst] = emb.squeeze(0).cpu()  

        del temp_clip_model

        cache_data = {
            "train_files": train_files,
            "val_files": val_files,
            "state_norm_dict": state_normalizer.state_dict(),
            "action_norm_dict": action_normalizer.state_dict(),
            "text_embedding_cache": text_embedding_dict,
        }
        torch.save(cache_data, self.cache_path)
        print(f"[Rank 0] Data preparation completed. Cache saved.")

    def setup(self, stage=None):
        """Load datasets from cache. This is called on every process in distributed training."""
        cache_data = torch.load(self.cache_path, weights_only=False)

        self.state_norm = RamenNormalizer(state_dict=cache_data["state_norm_dict"])
        self.action_norm = RamenNormalizer(state_dict=cache_data["action_norm_dict"])
        text_cache = cache_data["text_embedding_cache"]

        if stage == "fit" or stage is None:
            self.train_dataset = MultiViewDataset(
                file_list=cache_data["train_files"],
                state_norm=self.state_norm,
                action_norm=self.action_norm,
                text_cache=text_cache,
            )
            self.val_dataset = MultiViewDataset(
                file_list=cache_data["val_files"],
                state_norm=self.state_norm,
                action_norm=self.action_norm,
                text_cache=text_cache,
            )

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
