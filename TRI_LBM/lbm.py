from __future__ import annotations
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor, cat
from torch.utils._pytree import tree_map
from torch.nn import Module

# ein notation
# b - batch
# t - time
# c - channels
# h - height
# w - width
# d - dimension
# na - num actions

import einx
from einops import rearrange, repeat

# dogfooding
from x_transformers import Encoder
from denoising_diffusion_pytorch import GaussianDiffusion1D

# open clip
import open_clip

# functions


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def compact(arr):
    return [*filter(exists, arr)]


def maybe_cat(arr, *, dim):
    if len(arr) == 0:
        return None

    return cat(arr, dim=dim)


def xnor(x, y):
    return not (x ^ y)


def l2norm(t):
    return F.normalize(t, dim=-1)


def detach_all(obj):
    return tree_map(lambda t: t.detach() if is_tensor(t) else t, obj)


def divisible_by(num, den):
    return (num % den) == 0


def inputs_to_module_device(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        assert hasattr(self, "device")
        device = self.device

        args, kwargs = tree_map(
            lambda t: t.to(device) if is_tensor(t) else t, (args, kwargs)
        )

        return fn(self, *args, **kwargs)

    return inner


# random sinusoidal for times - used by deepmind a lot
class RandomSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=False)

    def forward(self, x):
        freqs = einx.multiply("b, d -> b d", x, self.weights) * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


# DiT wrapper
class DiffusionTransformerWrapper(Module):
    def __init__(
        self,
        dim_input,
        transformer: Encoder,
        dim_global_cond,  # Dimension of concatenated [Text, Img, Pose]
        dim_hidden=768,
    ):
        super().__init__()

        self.transformer = transformer
        self.dim_hidden = dim_hidden

        self.proj_in = nn.Linear(dim_input, dim_hidden)
        self.proj_out = nn.Linear(dim_hidden, dim_input)

        # Diffusion Timestep Processing (Paper: Sinusoidal -> 2-layer MLP)
        self.to_time_cond = nn.Sequential(
            RandomSinusoidalPosEmb(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        # Global Condition Processing
        # Since we use CrossAttention to simulate conditioning, we project flat cond to hidden dim
        self.cond_proj = nn.Sequential(
            nn.Linear(dim_global_cond, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

    def forward(
        self,
        actions,
        times,
        global_cond_feats=None,
    ):
        if global_cond_feats is None:
            global_cond_feats = self._temp_cond_feats
        tokens = self.proj_in(actions)
        time_cond = self.to_time_cond(times)
        cond_emb = self.cond_proj(global_cond_feats)
        final_cond = cond_emb + time_cond

        attended = self.transformer(
            tokens,
            condition=final_cond,
        )

        pred = self.proj_out(attended)
        return pred


class LBM(Module):
    def __init__(
        self,
        action_dim,
        dim_pose,
        dim=768,  # Paper: embedding size of 768
        depth=8,  # Paper: eight DiT blocks
        dim_head=64,
        heads=12,
        action_chunk_length=16,
        diffusion_timesteps=100,
        diffusion_sampling_timesteps=100,
        clip_language_model="ViT-B-32",
        language_pretrained_name="laion2b_s34b_b79k",
        clip_image_model="ViT-B-16",
        image_pretrained_name="openai",
        num_image_frames=2,  # Paper: two timesteps
        num_cameras=2,  # Assuming 2 cameras based on task
    ):
        super().__init__()
        # Clip, they use
        # ViT-B-16 for images
        # ViT-B-32 for language
        language_model, _, _ = open_clip.create_model_and_transforms(
            clip_language_model, pretrained=language_pretrained_name
        )
        language_model.eval()
        for param in language_model.parameters():
            param.requires_grad = False  # Paper: Keep frozen
        tokenizer = open_clip.get_tokenizer(clip_language_model)
        self.language_model = language_model
        self.language_tokenizer = tokenizer

        image_model, _, _ = open_clip.create_model_and_transforms(
            clip_image_model, pretrained=image_pretrained_name
        )

        # cheap way to get feat dimensions
        # assume one image for starters
        dim_text_feats = language_model.encode_text(tokenizer(["test"])).shape[-1]
        dim_image_feats = image_model.encode_image(torch.randn(1, 3, 224, 224)).shape[
            -1
        ]
        self.image_model = image_model
        self.text_proj = nn.Linear(dim_text_feats, dim)

        dim_per_step = dim + (num_cameras * dim_image_feats) + dim_pose
        self.dim_global_cond = num_image_frames * dim_per_step

        self.images_shape = (
            3,
            num_image_frames,
            224,
            224,
        )  # just enforce this shape to begin with

        self.diffusion_transformer = DiffusionTransformerWrapper(
            dim_input=action_dim,
            dim_global_cond=self.dim_global_cond,
            dim_hidden=dim,
            transformer=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dim_head=dim_head,
                cross_attend=False,
                use_adaptive_layernorm=True,
                rotary_pos_emb=True,
            ),
        )

        self.gaussian_diffusion_1d = GaussianDiffusion1D(
            self.diffusion_transformer,
            seq_length=action_chunk_length,
            timesteps=diffusion_timesteps,
            sampling_timesteps=diffusion_sampling_timesteps,
            channels=action_dim,
            auto_normalize=False,
            self_condition=False,
            channel_first=False,
        )

        self.num_image_frames = num_image_frames
        self.num_cameras = num_cameras
        self.dim_pose = dim_pose

        self.register_buffer("dummy", tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    @inputs_to_module_device
    def get_structured_features(
        self, text: list[str] | Tensor, images: Tensor, pose: Tensor
    ):
        """
        Extracts and flattens features: Text, Images, Pose.
        Structure: [B, T * (Text + Images + Pose)]
        """
        # Text Features
        if is_tensor(text):
            text_embeds = text.to(self.device).float()
        else:
            text_tokens = self.language_tokenizer(text).to(self.device)
            with torch.no_grad():
                text_embeds = self.language_model.encode_text(text_tokens)

        # Trainable Projection (Paper requirement)
        text_embeds = self.text_proj(text_embeds)  # [B, Dim]

        # Image Features (Multi-view + Multi-step)
        # images: [B, Nc, T, C, H, W] or [B, Nc, T, 3, 224, 224]
        b, nc, t, c, h, w = images.shape

        # We need to process each image independently through CLS
        # Flatten: (B * Nc * T), C, H, W
        flat_images = rearrange(images, "b nc t c h w -> (b nc t) c h w")

        # We want CLS tokens (encode_image usually returns global pooled/CLS)
        image_embeds = self.image_model.encode_image(flat_images)  # [B*Nc*T, 512]
        # Reshape back: [B, T, Nc, 512]
        image_embeds = rearrange(
            image_embeds, "(b nc t) d -> b t nc d", b=b, nc=nc, t=t
        )

        # Flatten cameras: [B, T, Nc*512]
        image_embeds = rearrange(image_embeds, "b t nc d -> b t (nc d)")
        # Proprioception (Pose)
        assert (
            pose.shape[1] == t
        ), f"Pose temporal dim {pose.shape[1]} != Image temporal dim {t}"
        # Concatenate Per Timestep
        text_embeds_seq = repeat(text_embeds, "b d -> b t d", t=t)
        # [B, T, (Text + Img + Pose)]
        per_step_features = torch.cat([text_embeds_seq, image_embeds, pose], dim=-1)

        # Flatten to Global Vector
        # [B, T * Feature_Width]
        global_cond = rearrange(per_step_features, "b t d -> b (t d)")

        return global_cond

    def _check_image_shape(self, images):
        # images: [B, Nc, T, C, H, W]
        assert (
            images.ndim == 6
        ), f"Expected 6D images [B, Nc, T, C, H, W], got {images.shape}"

        b, nc, t, c, h, w = images.shape

        assert nc == self.num_cameras, f"Expected {self.num_cameras} cameras, got {nc}"
        assert (
            t == self.num_image_frames
        ), f"Expected {self.num_image_frames} observation frames, got {t}"
        assert c == 3, f"Expected 3 channels (RGB), got {c}"
        assert (
            h == 224 and w == 224
        ), f"Expected 224x224 resolution for CLIP, got {h}x{w}"

    @inputs_to_module_device
    @torch.no_grad()
    def sample(
        self,
        text: list[str] | Tensor,
        images: Tensor,
        pose: Tensor,
    ):
        self._check_image_shape(images)
        global_cond = self.get_structured_features(text, images, pose)
        batch_size = images.shape[0]

        # sample actions
        self.diffusion_transformer._temp_cond_feats = global_cond
        sampled_actions = self.gaussian_diffusion_1d.sample(
            batch_size=batch_size,
        )
        self.diffusion_transformer._temp_cond_feats = None

        return sampled_actions

    @inputs_to_module_device
    def forward(
        self,
        text: list[str] | Tensor,
        images: Tensor,
        pose: Tensor,
        actions: Tensor | None = None,
    ):
        if not exists(actions):
            return self.sample(text, images, pose)
        self._check_image_shape(images)

        # Extract Global Condition
        global_cond = self.get_structured_features(text, images, pose)

        # Diffusion loss
        loss = self.gaussian_diffusion_1d(
            actions,
            model_forward_kwargs=dict(global_cond_feats=global_cond),
            return_reduced_loss=False,
        )
        loss = loss.mean()

        return loss
