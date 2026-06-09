"""
CatVTON Fine-tuning Script
==========================
Fine-tunes the CatVTON attention adapter on paired (person, garment, ground-truth)
data with optional DensePose IUV conditioning and mask-weighted loss.

Only the self-attention adapter modules are trained; the base UNet, VAE, and
scheduler remain frozen.  When IUV conditioning is enabled, the IUVEncoder
and the patched UNet conv_in layer are also trained.

Usage
-----
Single GPU:
    python train.py \\
        --data_root /path/to/dataset \\
        --output_dir ./checkpoints/catvton-finetune \\
        --use_iuv_conditioning

Multi-GPU (accelerate):
    accelerate launch train.py \\
        --data_root /path/to/dataset \\
        --output_dir ./checkpoints/catvton-finetune \\
        --use_iuv_conditioning \\
        --train_batch_size 2

Dataset format
--------------
The dataset directory must contain:
    <data_root>/
        person/   *.jpg / *.png   — person images
        garment/  *.jpg / *.png   — corresponding garment images (same filename)
        gt/       *.jpg / *.png   — ground-truth try-on result (same filename)
        mask/     *.png           — binary inpainting mask (same filename, optional)

If mask/ is absent, AutoMasker generates masks on-the-fly (slower).
If use_iuv_conditioning is set, DensePose IUV maps are computed on-the-fly
from person images.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import snapshot_download
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from model.attn_processor import AttnProcessor2_0
from model.iuv_encoder import IUVEncoder, IUV_LATENT_CHANNELS, IUV_IN_CHANNELS, prepare_iuv_latent
from model.pipeline import CatVTONPipeline
from model.utils import get_trainable_module, init_adapter
from utils import (
    center_garment,
    compute_mask_weighted_loss,
    compute_vae_encodings,
    init_accelerator,
    init_weight_dtype,
    prepare_image,
    prepare_mask_image,
    resize_and_crop,
    resize_and_padding,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass (mirrors argparse namespace)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Paths
    data_root: str = "./data"
    output_dir: str = "./checkpoints/catvton-finetune"
    base_ckpt: str = "runwayml/stable-diffusion-inpainting"
    attn_ckpt: str = "zhengchong/CatVTON"
    attn_ckpt_version: str = "mix"
    densepose_ckpt: str = ""
    schp_ckpt: str = ""

    # Image
    height: int = 512
    width: int = 384

    # Training
    train_batch_size: int = 1
    num_train_epochs: int = 10
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Loss
    mask_loss_weight: float = 5.0   # weight for pixels inside the inpainting mask

    # IUV conditioning
    use_iuv_conditioning: bool = False

    # Misc
    mixed_precision: str = "fp16"   # "no" | "fp16" | "bf16"
    report_to: str = "tensorboard"
    project_name: str = "catvton-finetune"
    save_steps: int = 500
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TryOnDataset(Dataset):
    """
    Paired virtual try-on dataset.

    Expects:
        <root>/person/   — person images
        <root>/garment/  — garment images (same filename as person)
        <root>/gt/       — ground-truth result images (same filename)
        <root>/mask/     — binary masks (optional, same filename, .png)
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        root: str,
        height: int = 512,
        width: int = 384,
        use_iuv_conditioning: bool = False,
        densepose_ckpt: str = "",
        device: str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.height = height
        self.width = width
        self.use_iuv_conditioning = use_iuv_conditioning

        # Collect filenames from person/ directory
        person_dir = self.root / "person"
        self.filenames = sorted([
            p.name for p in person_dir.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        ])
        assert len(self.filenames) > 0, f"No images found in {person_dir}"

        # Lazy-load DensePose only if IUV conditioning is requested
        self._densepose = None
        if use_iuv_conditioning and densepose_ckpt:
            from model.DensePose import DensePose
            self._densepose = DensePose(model_path=densepose_ckpt, device=device)

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_image(self, subdir: str, name: str) -> Image.Image:
        path = self.root / subdir / name
        # Try exact name first, then swap extension to .png
        if not path.exists():
            stem = Path(name).stem
            for ext in self.EXTENSIONS:
                alt = self.root / subdir / (stem + ext)
                if alt.exists():
                    path = alt
                    break
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> dict:
        name = self.filenames[idx]

        person_img  = self._load_image("person",  name)
        garment_img = self._load_image("garment", name)
        gt_img      = self._load_image("gt",      name)

        # ── Resize with consistent strategy ──────────────────────────────────
        # Person: center crop to target (preserves aspect ratio, no distortion)
        person_img = resize_and_crop(person_img, (self.width, self.height))
        # Garment: center content, then letterbox pad (keeps full garment visible)
        garment_img = center_garment(garment_img)
        garment_img = resize_and_padding(garment_img, (self.width, self.height))
        # GT: same as person (center crop)
        gt_img = resize_and_crop(gt_img, (self.width, self.height))

        # ── Mask: resize exactly like person (center crop + nearest) ─────────
        mask_path = self.root / "mask" / (Path(name).stem + ".png")
        if mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            # Apply same center-crop logic as person image, with NEAREST resampling
            mask_img = resize_and_crop(mask_img, (self.width, self.height), resample=Image.NEAREST)
            mask_arr = np.array(mask_img).astype(np.float32) / 255.0
        else:
            # Fallback: mask the entire image (model learns from scratch)
            mask_arr = np.ones((self.height, self.width), dtype=np.float32)

        # Binarize mask strictly
        mask_arr = (mask_arr >= 0.5).astype(np.float32)

        # ── Convert to tensors (normalized to [-1, 1] for images) ────────────
        person_t  = prepare_image(person_img)   # (1, 3, H, W) in [-1, 1]
        garment_t = prepare_image(garment_img)  # (1, 3, H, W) in [-1, 1]
        gt_t      = prepare_image(gt_img)       # (1, 3, H, W) in [-1, 1]
        mask_t    = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) binary

        sample = {
            "person":  person_t.squeeze(0),   # (3, H, W)
            "garment": garment_t.squeeze(0),
            "gt":      gt_t.squeeze(0),
            "mask":    mask_t.squeeze(0),      # (1, H, W)
        }

        # IUV map (computed on-the-fly if DensePose is available)
        if self.use_iuv_conditioning and self._densepose is not None:
            iuv = self._densepose.call_iuv(person_img, resize=max(self.height, self.width))
            # iuv: (H_orig, W_orig, 26) — resize to (height, width)
            import cv2
            n_ch = iuv.shape[2]  # 26
            iuv_resized = np.stack([
                cv2.resize(
                    iuv[:, :, c],
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST if c < 24 else cv2.INTER_LINEAR,
                )
                for c in range(n_ch)
            ], axis=2)
            sample["iuv"] = torch.from_numpy(iuv_resized).permute(2, 0, 1)  # (26, H, W)
        else:
            sample["iuv"] = torch.zeros(IUV_IN_CHANNELS, self.height, self.width)  # (26, H, W)

        return sample


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune CatVTON attention adapter")

    parser.add_argument("--data_root",          type=str,   default="./data")
    parser.add_argument("--output_dir",         type=str,   default="./checkpoints/catvton-finetune")
    parser.add_argument("--base_ckpt",          type=str,   default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--attn_ckpt",          type=str,   default="zhengchong/CatVTON")
    parser.add_argument("--attn_ckpt_version",  type=str,   default="mix")
    parser.add_argument("--densepose_ckpt",     type=str,   default="")
    parser.add_argument("--schp_ckpt",          type=str,   default="")
    parser.add_argument("--height",             type=int,   default=512)
    parser.add_argument("--width",              type=int,   default=384)
    parser.add_argument("--train_batch_size",   type=int,   default=1)
    parser.add_argument("--num_train_epochs",   type=int,   default=10)
    parser.add_argument("--max_train_steps",    type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate",      type=float, default=1e-5)
    parser.add_argument("--lr_scheduler",       type=str,   default="cosine")
    parser.add_argument("--lr_warmup_steps",    type=int,   default=500)
    parser.add_argument("--mask_loss_weight",   type=float, default=5.0)
    parser.add_argument("--use_iuv_conditioning", action="store_true")
    parser.add_argument("--mixed_precision",    type=str,   default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to",          type=str,   default="tensorboard")
    parser.add_argument("--project_name",       type=str,   default="catvton-finetune")
    parser.add_argument("--save_steps",         type=int,   default=500)
    parser.add_argument("--seed",               type=int,   default=42)

    args = parser.parse_args()
    cfg = TrainConfig(**{k: v for k, v in vars(args).items() if k in TrainConfig.__dataclass_fields__})
    return cfg


def main() -> None:
    cfg = parse_args()

    # ── Accelerator ─────────────────────────────────────────────────────────
    accelerator = init_accelerator(cfg)
    weight_dtype = init_weight_dtype(cfg.mixed_precision)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)

    # ── Models ──────────────────────────────────────────────────────────────
    logger.info("Loading models...")

    noise_scheduler = DDIMScheduler.from_pretrained(cfg.base_ckpt, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
        accelerator.device, dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(cfg.base_ckpt, subfolder="unet").to(
        accelerator.device, dtype=weight_dtype
    )

    # Patch UNet conv_in for IUV channels if needed
    if cfg.use_iuv_conditioning:
        CatVTONPipeline._patch_unet_conv_in(unet, extra_channels=IUV_LATENT_CHANNELS)

    # Replace cross-attention with AttnProcessor2_0 (standard cross-attention)
    init_adapter(unet, cross_attn_cls=AttnProcessor2_0)

    # Load pretrained attention adapter
    attn_modules = get_trainable_module(unet, "attention")
    sub_folder = {
        "mix":       "mix-48k-1024",
        "vitonhd":   "vitonhd-16k-512",
        "dresscode": "dresscode-16k-512",
    }[cfg.attn_ckpt_version]
    if os.path.exists(cfg.attn_ckpt):
        ckpt_path = os.path.join(cfg.attn_ckpt, sub_folder, "attention")
    else:
        repo_path = snapshot_download(repo_id=cfg.attn_ckpt)
        ckpt_path = os.path.join(repo_path, sub_folder, "attention")
    from accelerate import load_checkpoint_in_model
    load_checkpoint_in_model(attn_modules, ckpt_path)
    logger.info("Loaded attention adapter from %s", ckpt_path)

    # IUV encoder
    iuv_encoder: Optional[IUVEncoder] = None
    if cfg.use_iuv_conditioning:
        iuv_encoder = IUVEncoder(
            in_channels=IUV_IN_CHANNELS,       # 26: 24 one-hot I + U + V
            mid_channels=32,
            out_channels=IUV_LATENT_CHANNELS,  # 8
        ).to(accelerator.device, dtype=weight_dtype)

    # ── CLIP vision encoder (frozen, garment image conditioning) ──────────
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(accelerator.device, dtype=weight_dtype)
    clip_image_encoder.requires_grad_(False)

    logger.info("CLIP vision encoder loaded (openai/clip-vit-large-patch14)")

    # ── Freeze / unfreeze ───────────────────────────────────────────────────
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Trainable: attention adapter
    attn_modules.requires_grad_(True)

    # Trainable: IUV encoder + patched conv_in (when IUV conditioning is on)
    trainable_params = list(attn_modules.parameters())
    if cfg.use_iuv_conditioning:
        iuv_encoder.requires_grad_(True)
        unet.conv_in.requires_grad_(True)
        trainable_params += list(iuv_encoder.parameters())
        trainable_params += list(unet.conv_in.parameters())

    n_params = sum(p.numel() for p in trainable_params)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # ── Dataset & DataLoader ────────────────────────────────────────────────
    dataset = TryOnDataset(
        root=cfg.data_root,
        height=cfg.height,
        width=cfg.width,
        use_iuv_conditioning=cfg.use_iuv_conditioning,
        densepose_ckpt=cfg.densepose_ckpt,
        device=str(accelerator.device),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ── LR Scheduler ────────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.max_train_steps or cfg.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=max_train_steps * cfg.gradient_accumulation_steps,
    )

    # ── Accelerate prepare ──────────────────────────────────────────────────
    if cfg.use_iuv_conditioning:
        unet, iuv_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, iuv_encoder, optimizer, dataloader, lr_scheduler
        )
    else:
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

    # ── Training loop ───────────────────────────────────────────────────────
    global_step = 0
    concat_dim = -2  # spatial y-axis concat (same as pipeline)

    logger.info("Starting training — %d steps", max_train_steps)

    for epoch in range(cfg.num_train_epochs):
        unet.train()
        if iuv_encoder is not None:
            iuv_encoder.train()

        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}",
        )

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):

                # ── Encode images to latents ─────────────────────────────
                # person (masked) latent
                person   = batch["person"].to(accelerator.device, dtype=weight_dtype)   # (B, 3, H, W)
                garment  = batch["garment"].to(accelerator.device, dtype=weight_dtype)
                gt       = batch["gt"].to(accelerator.device, dtype=weight_dtype)
                mask     = batch["mask"].to(accelerator.device, dtype=weight_dtype)     # (B, 1, H, W)

                masked_person = person * (mask < 0.5)

                with torch.no_grad():
                    masked_latent   = compute_vae_encodings(masked_person, vae)   # (B, 4, H/8, W/8)
                    garment_latent  = compute_vae_encodings(garment, vae)
                    gt_latent       = compute_vae_encodings(gt, vae)

                # Downscale mask to latent resolution (nearest to preserve binary edges)
                mask_latent = F.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")

                # ── IUV latent ───────────────────────────────────────────
                if cfg.use_iuv_conditioning and iuv_encoder is not None:
                    iuv = batch["iuv"].to(accelerator.device, dtype=weight_dtype)  # (B, 26, H, W)
                    iuv_latent = iuv_encoder(iuv)                                   # (B, 8, H/8, W/8)
                else:
                    iuv_latent = torch.zeros(
                        masked_latent.shape[0], IUV_LATENT_CHANNELS,
                        masked_latent.shape[-2], masked_latent.shape[-1],
                        device=accelerator.device, dtype=weight_dtype,
                    )

                # ── Spatial concat (y-axis) ──────────────────────────────
                # [person_masked | garment] stacked vertically
                masked_latent_concat = torch.cat([masked_latent, garment_latent], dim=concat_dim)
                mask_latent_concat   = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
                gt_latent_concat     = torch.cat([gt_latent, torch.zeros_like(gt_latent)], dim=concat_dim)

                # ── Add noise ────────────────────────────────────────────
                noise = torch.randn_like(gt_latent_concat)
                bsz   = gt_latent_concat.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=accelerator.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(gt_latent_concat, noise, timesteps)

                # ── Build UNet input ─────────────────────────────────────
                # Channels: [noisy(4), mask(1), masked_image(4), iuv(8)] = 17
                # (or 9 without IUV conditioning)
                # IUV latent is tiled to match the doubled spatial height
                iuv_latent_tiled = torch.cat([iuv_latent, torch.zeros_like(iuv_latent)], dim=concat_dim)
                model_input = torch.cat(
                    [noisy_latents, mask_latent_concat, masked_latent_concat, iuv_latent_tiled],
                    dim=1,
                )

                # ── UNet forward ─────────────────────────────────────────
                # Encode garment images with CLIP vision for cross-attention
                # garment tensor is (B, 3, H, W) in [-1, 1], convert to PIL for CLIP
                with torch.no_grad():
                    # Convert garment tensor to uint8 numpy for CLIPImageProcessor
                    garment_np = ((garment.cpu().float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
                    garment_pil_list = [
                        Image.fromarray(garment_np[b].permute(1, 2, 0).numpy())
                        for b in range(garment_np.shape[0])
                    ]
                    clip_inputs = clip_processor(
                        images=garment_pil_list,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    clip_inputs["pixel_values"] = clip_inputs["pixel_values"].to(dtype=weight_dtype)
                    garment_embeds = clip_image_encoder(**clip_inputs).image_embeds  # (B, proj_dim)
                    # Reshape for encoder_hidden_states: (B, 1, proj_dim)
                    encoder_hidden_states = garment_embeds.unsqueeze(1)

                noise_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

                # ── Mask-weighted loss ───────────────────────────────────
                # Use only the person-half of the spatial concat for loss
                # (top half = person region, bottom half = garment region)
                half_h = noise_pred.shape[concat_dim] // 2
                noise_pred_person = noise_pred.split(half_h, dim=concat_dim)[0]
                noise_target      = noise.split(half_h, dim=concat_dim)[0]

                # Diffusion loss (mask-weighted MSE) — unchanged
                loss = compute_mask_weighted_loss(
                    noise_pred_person,
                    noise_target,
                    mask_latent,
                    mask_weight=cfg.mask_loss_weight,
                )

                # Masked L1 loss — penalises prediction error inside the
                # inpainting region in absolute terms.
                # mask_latent is (B, 1, H/8, W/8); broadcasts across the
                # 4 latent channels of noise_pred_person automatically.
                # iuv_latent is NOT used here — this loss operates purely
                # on the diffusion prediction vs. the noise target.
                masked_l1 = F.l1_loss(
                    noise_pred_person * mask_latent,
                    noise_target      * mask_latent,
                )
                loss = loss + 0.2 * masked_l1

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── Logging & checkpointing ──────────────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    accelerator.log({
                        "train/loss":      loss.detach().item(),
                        "train/masked_l1": masked_l1.detach().item(),
                    }, step=global_step)

                    if global_step % cfg.save_steps == 0:
                        save_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)

                        # Save attention adapter
                        attn_save_dir = os.path.join(save_dir, "attention")
                        os.makedirs(attn_save_dir, exist_ok=True)
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        attn_state = get_trainable_module(unwrapped_unet, "attention").state_dict()
                        torch.save(attn_state, os.path.join(attn_save_dir, "pytorch_model.bin"))

                        # Save IUV encoder if used
                        if iuv_encoder is not None:
                            iuv_state = accelerator.unwrap_model(iuv_encoder).state_dict()
                            torch.save(iuv_state, os.path.join(save_dir, "iuv_encoder.bin"))

                        # Save patched conv_in if used
                        if cfg.use_iuv_conditioning:
                            conv_state = unwrapped_unet.conv_in.state_dict()
                            torch.save(conv_state, os.path.join(save_dir, "unet_conv_in.bin"))

                        logger.info("Saved checkpoint to %s", save_dir)

            if global_step >= max_train_steps:
                break

        progress_bar.close()
        if global_step >= max_train_steps:
            break

    # ── Final save ──────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(unet)
        attn_save_dir = os.path.join(final_dir, "attention")
        os.makedirs(attn_save_dir, exist_ok=True)
        attn_state = get_trainable_module(unwrapped_unet, "attention").state_dict()
        torch.save(attn_state, os.path.join(attn_save_dir, "pytorch_model.bin"))

        if iuv_encoder is not None:
            iuv_state = accelerator.unwrap_model(iuv_encoder).state_dict()
            torch.save(iuv_state, os.path.join(final_dir, "iuv_encoder.bin"))

        if cfg.use_iuv_conditioning:
            conv_state = unwrapped_unet.conv_in.state_dict()
            torch.save(conv_state, os.path.join(final_dir, "unet_conv_in.bin"))

        logger.info("Training complete. Final checkpoint saved to %s", final_dir)

    accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
