import inspect
import os
from typing import Optional, Union

import PIL
import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from model.attn_processor import AttnProcessor2_0
from model.iuv_encoder import IUVEncoder, IUV_IN_CHANNELS, prepare_iuv_latent
from model.utils import get_trainable_module, init_adapter
from utils import (center_garment, compute_vae_encodings, numpy_to_pil,
                    prepare_image, prepare_mask_image, preprocess_inputs,
                    resize_and_crop, resize_and_padding)

# Number of IUV latent channels added to the UNet input.
# Must match IUVEncoder(out_channels=IUV_LATENT_CHANNELS).
IUV_LATENT_CHANNELS = 8


class CatVTONPipeline:
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt, 
        attn_ckpt_version="mix",
        weight_dtype=torch.float16,
        device='cuda',
        compile=True,
        skip_safety_check=False,
        use_tf32=True,
        use_iuv_conditioning=False,   # NEW: enable DensePose IUV conditioning
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check
        self.use_iuv_conditioning = use_iuv_conditioning

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)

        # ── CLIP vision encoder (garment image conditioning) ─────────────────
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device, dtype=weight_dtype)
        self.clip_image_encoder.requires_grad_(False)
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=weight_dtype)

        # ── IUV encoder (optional) ──────────────────────────────────────────
        # When enabled, the UNet receives IUV_LATENT_CHANNELS extra input
        # channels (concatenated along dim=1 after the standard 9 channels).
        # The UNet's conv_in layer is patched to accept the wider input.
        if use_iuv_conditioning:
            self.iuv_encoder = IUVEncoder(
                in_channels=IUV_IN_CHANNELS,   # 26: 24 one-hot I + U + V
                mid_channels=32,
                out_channels=IUV_LATENT_CHANNELS,  # 8
            ).to(device, dtype=weight_dtype)
        else:
            self.iuv_encoder = None

        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)

        # Patch UNet conv_in to accept extra IUV channels if needed
        if use_iuv_conditioning:
            self._patch_unet_conv_in(self.unet, extra_channels=IUV_LATENT_CHANNELS)

        init_adapter(self.unet, cross_attn_cls=AttnProcessor2_0)  # Standard cross-attention with CLIP embeddings
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")
            
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    @staticmethod
    def _patch_unet_conv_in(unet: UNet2DConditionModel, extra_channels: int) -> None:
        """
        Expand the UNet's first conv layer (conv_in) to accept extra input
        channels from the IUV encoder.

        The original conv_in has in_channels = 9 (for SD-inpainting):
            4 noisy latent + 1 mask + 4 masked-image latent
        After patching it becomes 9 + extra_channels.

        The existing weights are preserved; the new channels are initialised
        to zero so the model starts as if IUV conditioning is absent.
        """
        old_conv = unet.conv_in
        old_in_ch = old_conv.in_channels
        new_in_ch = old_in_ch + extra_channels

        new_conv = torch.nn.Conv2d(
            new_in_ch,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        # Copy existing weights; zero-init the new channels
        with torch.no_grad():
            new_conv.weight[:, :old_in_ch] = old_conv.weight
            new_conv.weight[:, old_in_ch:] = 0.0
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        unet.conv_in = new_conv
        unet.config["in_channels"] = new_in_ch

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
        return image, has_nsfw_concept
    
    def _encode_garment(self, garment_image: PIL.Image.Image) -> torch.Tensor:
        """
        Encode a garment PIL image into CLIP vision embeddings for UNet cross-attention.

        Returns:
            (1, 1, projection_dim) tensor suitable for encoder_hidden_states.
        """
        inputs = self.clip_processor(
            images=garment_image,
            return_tensors="pt",
        ).to(self.device)
        # Move pixel_values to the correct dtype
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.weight_dtype)
        embeds = self.clip_image_encoder(**inputs).image_embeds  # (1, projection_dim)
        return embeds.unsqueeze(1)  # (1, 1, projection_dim)

    def check_inputs(self, image, condition_image, mask, width, height):
        """
        Validate and preprocess inputs. If inputs are already tensors, return as-is.
        Otherwise, run the full preprocessing pipeline.

        Returns:
            image_tensor:   (1, 3, H, W) in [-1, 1]
            garment_tensor: (1, 3, H, W) in [-1, 1]
            mask_tensor:    (1, 1, H, W) binary {0, 1}
            garment_pil:    PIL Image of resized garment (for CLIP encoding)
        """
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            # Already tensors — ensure correct shapes
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if condition_image.ndim == 3:
                condition_image = condition_image.unsqueeze(0)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(0)
            # Binarize mask
            mask = (mask >= 0.5).float()
            # Convert garment tensor to PIL for CLIP
            garment_np = ((condition_image[0].permute(1, 2, 0).cpu().float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
            garment_pil = PIL.Image.fromarray(garment_np)
            return image, condition_image, mask, garment_pil

        # PIL inputs — use unified preprocessing
        return preprocess_inputs(image, condition_image, mask, height, width)
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 24,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        iuv_map: Optional[np.ndarray] = None,  # (H, W, 3) float32 IUV from DensePose.call_iuv()
        **kwargs
    ):
        concat_dim = -2  # y axis concat

        # ── Preprocessing ───────────────────────────────────────────────────
        # Unified input handling: resize, normalize, binarize mask
        image_t, condition_t, mask_t, garment_pil = self.check_inputs(
            image, condition_image, mask, width, height
        )
        image_t = image_t.to(self.device, dtype=self.weight_dtype)
        condition_t = condition_t.to(self.device, dtype=self.weight_dtype)
        mask_t = mask_t.to(self.device, dtype=self.weight_dtype)

        # Encode garment with CLIP vision
        garment_embeds = self._encode_garment(garment_pil)  # (1, 1, proj_dim)

        # Mask person image
        masked_image = image_t * (mask_t < 0.5)

        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_t, self.vae)
        # Downsample mask to latent resolution with nearest interpolation (preserve binary)
        mask_latent = torch.nn.functional.interpolate(
            mask_t, size=masked_latent.shape[-2:], mode="nearest"
        )
        del image_t, mask_t, condition_t

        # ── IUV conditioning ────────────────────────────────────────────────
        # Encode IUV map to latent-resolution feature tensor and concatenate
        # to the UNet input along the channel dimension (dim=1).
        # iuv_latent shape: (1, IUV_LATENT_CHANNELS, H/8, W/8)
        if self.use_iuv_conditioning and iuv_map is not None:
            iuv_latent = prepare_iuv_latent(
                iuv_map, height, width,
                self.iuv_encoder, self.device, self.weight_dtype,
            )
        else:
            # Zero tensor — no IUV conditioning (backward-compatible)
            iuv_latent = torch.zeros(
                1, IUV_LATENT_CHANNELS,
                masked_latent.shape[-2], masked_latent.shape[-1],
                device=self.device, dtype=self.weight_dtype,
            )

        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)
            # Duplicate IUV latent for CFG (unconditional uses zeros)
            iuv_latent_cfg = torch.cat([torch.zeros_like(iuv_latent), iuv_latent], dim=0)
        else:
            iuv_latent_cfg = iuv_latent

        # ── CLIP garment embeddings for cross-attention (CFG) ────────────────
        if do_classifier_free_guidance:
            # CFG: [unconditional (zeros), conditional (garment)]
            garment_embeds = torch.cat([torch.zeros_like(garment_embeds), garment_embeds], dim=0)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
                # prepare the input for the inpainting model
                # Standard channels: [noisy_latent(4), mask(1), masked_image(4)] = 9 channels
                # + IUV channels: IUV_LATENT_CHANNELS extra channels
                inpainting_latent_model_input = torch.cat(
                    [non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat, iuv_latent_cfg],
                    dim=1,
                )
                # predict the noise residual
                noise_pred= self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=garment_embeds,
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        
        # Safety Check
        if not self.skip_safety_check:
            current_script_directory = os.path.dirname(os.path.realpath(__file__))
            nsfw_image = os.path.join(os.path.dirname(current_script_directory), 'resource', 'img', 'NSFW.jpg')
            nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
            image_np = np.array(image)
            _, has_nsfw_concept = self.run_safety_checker(image=image_np)
            for i, not_safe in enumerate(has_nsfw_concept):
                if not_safe:
                    image[i] = nsfw_image
        return image


class CatVTONPix2PixPipeline(CatVTONPipeline):
    def auto_attn_ckpt_load(self, attn_ckpt, version):
        # TODO: Temperal fix for the model version
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, version, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, version, 'attention'))
    
    def check_inputs(self, image, condition_image, width, height):
        """Validate and preprocess inputs for pix2pix pipeline."""
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if condition_image.ndim == 3:
                condition_image = condition_image.unsqueeze(0)
            garment_np = ((condition_image[0].permute(1, 2, 0).cpu().float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
            garment_pil = PIL.Image.fromarray(garment_np)
            return image, condition_image, garment_pil

        image = resize_and_crop(image, (width, height))
        condition_image_centered = center_garment(condition_image)
        condition_image_padded = resize_and_padding(condition_image_centered, (width, height))
        garment_pil = condition_image_padded.copy()

        image_t = prepare_image(image)
        condition_t = prepare_image(condition_image_padded)
        return image_t, condition_t, garment_pil

    @torch.no_grad()
    def __call__(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 24,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        concat_dim = -1

        # ── Preprocessing ───────────────────────────────────────────────────
        image_t, condition_t, garment_pil = self.check_inputs(
            image, condition_image, width, height
        )
        image_t = image_t.to(self.device, dtype=self.weight_dtype)
        condition_t = condition_t.to(self.device, dtype=self.weight_dtype)

        # Encode garment with CLIP vision
        garment_embeds = self._encode_garment(garment_pil)  # (1, 1, proj_dim)

        # VAE encoding
        image_latent = compute_vae_encodings(image_t, self.vae)
        condition_latent = compute_vae_encodings(condition_t, self.vae)
        del image_t, condition_t
        # Concatenate latents
        condition_latent_concat = torch.cat([image_latent, condition_latent], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            condition_latent_concat.shape,
            generator=generator,
            device=condition_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            condition_latent_concat = torch.cat(
                [
                    torch.cat([image_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    condition_latent_concat,
                ]
            )

        # ── CLIP garment embeddings for cross-attention (CFG) ────────────────
        if do_classifier_free_guidance:
            garment_embeds = torch.cat([torch.zeros_like(garment_embeds), garment_embeds], dim=0)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                # prepare the input for the inpainting model
                p2p_latent_model_input = torch.cat([latent_model_input, condition_latent_concat], dim=1)
                # predict the noise residual
                noise_pred= self.unet(
                    p2p_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=garment_embeds,
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        
        # Safety Check
        if not self.skip_safety_check:
            current_script_directory = os.path.dirname(os.path.realpath(__file__))
            nsfw_image = os.path.join(os.path.dirname(current_script_directory), 'resource', 'img', 'NSFW.jpg')
            nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
            image_np = np.array(image)
            _, has_nsfw_concept = self.run_safety_checker(image=image_np)
            for i, not_safe in enumerate(has_nsfw_concept):
                if not_safe:
                    image[i] = nsfw_image
        return image
