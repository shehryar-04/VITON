import os

import math
import PIL
import numpy as np
import torch
from PIL import Image
from accelerate.state import AcceleratorState
from packaging import version
import accelerate
from typing import List, Optional, Tuple, Set
from diffusers import UNet2DConditionModel, SchedulerMixin
from tqdm import tqdm


# Compute DREAM and update latents for diffusion sampling
def compute_dream_and_update_latents_for_inpaint(
    unet: UNet2DConditionModel,
    noise_scheduler: SchedulerMixin,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = None  # b, 4, h, w
    with torch.no_grad():
        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    noisy_latents_no_condition = noisy_latents[:, :4]
    _noisy_latents, _target = (None, None)
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        _noisy_latents = noisy_latents_no_condition.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        _target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    _noisy_latents = torch.cat([_noisy_latents, noisy_latents[:, 4:]], dim=1)
    return _noisy_latents, _target

# Prepare the input for inpainting model.
def prepare_inpainting_input(
    noisy_latents: torch.Tensor, 
    mask_latents: torch.Tensor,
    condition_latents: torch.Tensor,
    enable_condition_noise: bool = True,
    condition_concat_dim: int = -1,
) -> torch.Tensor:
    """
    Prepare the input for inpainting model.
    
    Args:
        noisy_latents (torch.Tensor): Noisy latents.
        mask_latents (torch.Tensor): Mask latents.
        condition_latents (torch.Tensor): Condition latents.
        enable_condition_noise (bool): Enable condition noise.
    
    Returns:
        torch.Tensor: Inpainting input.
    """
    if not enable_condition_noise:
        condition_latents_ = condition_latents.chunk(2, dim=condition_concat_dim)[-1]
        noisy_latents = torch.cat([noisy_latents, condition_latents_], dim=condition_concat_dim)
    noisy_latents = torch.cat([noisy_latents, mask_latents, condition_latents], dim=1)
    return noisy_latents

# Mask-weighted diffusion loss
def compute_mask_weighted_loss(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    mask_latent: torch.Tensor,
    mask_weight: float = 5.0,
) -> torch.Tensor:
    """
    Compute MSE diffusion loss with higher weight inside the inpainting mask.

    Pixels inside the mask (where the garment should appear) are weighted
    `mask_weight` times more than background pixels.  This focuses the model
    on getting the garment region right.

    Args:
        noise_pred   : (B, C, H, W) — UNet noise prediction
        noise_target : (B, C, H, W) — ground-truth noise (epsilon target)
        mask_latent  : (B, 1, H, W) — binary mask at latent resolution (1=masked)
        mask_weight  : scalar multiplier for masked pixels (default 5.0)

    Returns:
        Scalar loss tensor.

    Example::

        loss = compute_mask_weighted_loss(noise_pred, noise, mask_latent, mask_weight=5.0)
        accelerator.backward(loss)
    """
    # Per-pixel squared error: (B, C, H, W)
    per_pixel_loss = (noise_pred.float() - noise_target.float()) ** 2

    # Build weight map: 1.0 everywhere, mask_weight inside the mask
    # mask_latent is (B, 1, H, W) — broadcast across channels
    weight_map = 1.0 + (mask_weight - 1.0) * mask_latent.float()  # (B, 1, H, W)

    # Weighted mean
    weighted_loss = (per_pixel_loss * weight_map).mean()
    return weighted_loss


# Compute VAE encodings
def compute_vae_encodings(image: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): image to be encoded
        vae (torch.nn.Module): vae model

    Returns:
        torch.Tensor: latent encoding of the image
    """
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input


# Init Accelerator
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration

def init_accelerator(config):
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.project_name,
        logging_dir=os.path.join(config.project_name, "logs"),
    )
    accelerator_ddp_config = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_ddp_config],
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
        
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name,
            config={
                "learning_rate": config.learning_rate,
                "train_batch_size": config.train_batch_size,
                "image_size": f"{config.width}x{config.height}",
            },
        )
        
    return accelerator


def init_weight_dtype(wight_dtype):
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[wight_dtype]


def init_add_item_id(config):
    return torch.tensor(
        [
            config.height,
            config.width * 2,
            0,
            0,
            config.height,
            config.width * 2,
        ]
    ).repeat(config.train_batch_size, 1)


def repaint_result(result, person_image, mask_image):
    result, person, mask = np.array(result), np.array(person_image), np.array(mask_image)
    # expand the mask to 3 channels & to 0~1
    mask = np.expand_dims(mask, axis=2)
    mask = mask / 255.0
    # mask for result, ~mask for person
    result_ = result * mask + person * (1 - mask)
    return Image.fromarray(result_.astype(np.uint8))


def prepare_image(image):
    """
    Convert a PIL image, numpy array, or tensor to a (1, 3, H, W) float32 tensor
    normalized to [-1, 1] (Stable Diffusion expected range).

    Normalization: pixel / 127.5 - 1.0  (equivalent to (pixel - 0.5) / 0.5 on [0,1])

    Input:
        PIL.Image.Image, np.ndarray (H, W, 3) uint8, or torch.Tensor (3, H, W) or (1, 3, H, W)
    Output:
        torch.Tensor (1, 3, H, W) in [-1, 1], float32
    """
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image


def prepare_mask_image(mask_image):
    """
    Convert a mask (PIL, numpy, or tensor) to a strictly binary (1, 1, H, W) float32 tensor.

    - Values >= 0.5 become 1.0, values < 0.5 become 0.0.
    - No bilinear interpolation is applied here; resizing should be done
      separately with mode="nearest" to preserve binary edges.

    Input:
        PIL.Image.Image (mode "L"), np.ndarray (H, W), or torch.Tensor
    Output:
        torch.Tensor (1, 1, H, W) float32, strictly binary {0.0, 1.0}
    """
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            mask_image = mask_image.unsqueeze(1)
        mask_image = mask_image.to(dtype=torch.float32)
    else:
        if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
            mask_image = [mask_image]
        if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
            mask_image = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
            )
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)
            mask_image = mask_image.astype(np.float32)
        mask_image = torch.from_numpy(mask_image).to(dtype=torch.float32)

    # Binarize strictly
    mask_image = (mask_image >= 0.5).float()
    return mask_image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def tensor_to_image(tensor: torch.Tensor):
    """
    Converts a torch tensor to PIL Image.
    """
    assert tensor.dim() == 3, "Input tensor should be 3-dimensional."
    assert tensor.dtype == torch.float32, "Input tensor should be float32."
    assert (
        tensor.min() >= 0 and tensor.max() <= 1
    ), "Input tensor should be in range [0, 1]."
    tensor = tensor.cpu()
    tensor = tensor * 255
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy().astype(np.uint8)
    image = Image.fromarray(tensor)
    return image


def concat_images(images: List[Image.Image], divider: int = 4, cols: int = 4):
    """
    Concatenates images horizontally and with
    """
    widths = [image.size[0] for image in images]
    heights = [image.size[1] for image in images]
    total_width = cols * max(widths)
    total_width += divider * (cols - 1)
    # `col` images each row
    rows = math.ceil(len(images) / cols)
    total_height = max(heights) * rows
    # add divider between rows
    total_height += divider * (len(heights) // cols - 1)

    # all black image
    concat_image = Image.new("RGB", (total_width, total_height), (0, 0, 0))

    x_offset = 0
    y_offset = 0
    for i, image in enumerate(images):
        concat_image.paste(image, (x_offset, y_offset))
        x_offset += image.size[0] + divider
        if (i + 1) % cols == 0:
            x_offset = 0
            y_offset += image.size[1] + divider

    return concat_image


def read_prompt_file(prompt_file: str):
    if prompt_file is not None and os.path.isfile(prompt_file):
        with open(prompt_file, "r") as sample_prompt_file:
            sample_prompts = sample_prompt_file.readlines()
            sample_prompts = [sample_prompt.strip() for sample_prompt in sample_prompts]
    else:
        sample_prompts = []
    return sample_prompts


def save_tensors_to_npz(tensors: torch.Tensor, paths: List[str]):
    assert len(tensors) == len(paths), "Length of tensors and paths should be the same!"
    for tensor, path in zip(tensors, paths):
        np.savez_compressed(path, latent=tensor.cpu().numpy())


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = (
        AcceleratorState().deepspeed_plugin
        if accelerate.state.is_initialized()
        else None
    )
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def is_xformers_available():
    try:
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            print(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                "please update xFormers to at least 0.0.17. "
                "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        return True
    except ImportError:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly"
        )



def center_garment(image: Image.Image, background_threshold: int = 240) -> Image.Image:
    """
    Auto-center a garment image by detecting the non-background bounding box
    and centering the garment content within the frame.

    If the garment already occupies most of the frame (>80% area), returns as-is.
    Background is detected as pixels where all RGB channels exceed `background_threshold`.

    Args:
        image: PIL RGB garment image.
        background_threshold: pixel value above which a pixel is considered background.

    Returns:
        PIL Image with garment centered in the original frame size.
    """
    img_arr = np.array(image)
    # Detect non-background pixels (not near-white)
    non_bg_mask = np.any(img_arr < background_threshold, axis=2)

    if not non_bg_mask.any():
        # Entirely background — return as-is
        return image

    # Find bounding box of non-background region
    rows = np.where(non_bg_mask.any(axis=1))[0]
    cols = np.where(non_bg_mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1] + 1
    left, right = cols[0], cols[-1] + 1

    content_h = bottom - top
    content_w = right - left
    img_h, img_w = img_arr.shape[:2]

    # If content already fills >80% of the frame, skip centering
    content_area_ratio = (content_h * content_w) / (img_h * img_w)
    if content_area_ratio > 0.80:
        return image

    # Crop to content, then paste centered on white canvas
    content = image.crop((left, top, right, bottom))
    centered = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    paste_x = (img_w - content_w) // 2
    paste_y = (img_h - content_h) // 2
    centered.paste(content, (paste_x, paste_y))
    return centered


def preprocess_inputs(
    image: "PIL.Image.Image",
    garment: "PIL.Image.Image",
    mask: "PIL.Image.Image",
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, "PIL.Image.Image"]:
    """
    Unified preprocessing for the CatVTON pipeline.

    Steps:
      1. Person image: resize with center crop to (width, height).
      2. Garment image: center the garment, then resize with padding (letterbox).
      3. Mask: resize EXACTLY like person image (center crop), using NEAREST interpolation.
      4. Normalize person + garment to [-1, 1] (SD expected range).
      5. Mask stays binary float32 {0.0, 1.0}.

    Args:
        image:   PIL RGB person image.
        garment: PIL RGB garment image.
        mask:    PIL "L" or "RGB" mask (white = inpaint region).
        height:  Target height in pixels.
        width:   Target width in pixels.

    Returns:
        image_tensor:   (1, 3, H, W) float32 in [-1, 1]
        garment_tensor: (1, 3, H, W) float32 in [-1, 1]
        mask_tensor:    (1, 1, H, W) float32 binary {0, 1}
        garment_pil:    Resized garment as PIL (for CLIP encoding)
    """
    # ── 1. Person image: center crop to target ───────────────────────────────
    image = resize_and_crop(image, (width, height))

    # ── 2. Garment: center content, then letterbox pad ───────────────────────
    garment = center_garment(garment)
    garment = resize_and_padding(garment, (width, height))
    garment_pil = garment.copy()  # Keep PIL copy for CLIP before tensor conversion

    # ── 3. Mask: resize exactly like person (center crop + nearest) ──────────
    if mask.mode != "L":
        mask = mask.convert("L")
    mask = resize_and_crop(mask, (width, height), resample=Image.NEAREST)

    # ── 4. Convert to tensors ────────────────────────────────────────────────
    # Person + garment: normalized to [-1, 1]
    image_tensor = prepare_image(image)      # (1, 3, H, W)
    garment_tensor = prepare_image(garment)  # (1, 3, H, W)

    # Mask: binary float32
    mask_tensor = prepare_mask_image(mask)   # (1, 1, H, W)

    return image_tensor, garment_tensor, mask_tensor, garment_pil


def resize_and_crop(image, size, resample=Image.LANCZOS):
    """
    Resize a PIL image to `size` (width, height) using center crop.

    Strategy:
      1. Compute the crop region that matches the target aspect ratio.
      2. Center-crop to that region.
      3. Resize to the exact target dimensions.

    This preserves aspect ratio before the final resize, avoiding distortion.

    Args:
        image: PIL Image (any mode).
        size: (width, height) target size.
        resample: PIL resampling filter. Use Image.NEAREST for masks,
                  Image.LANCZOS for RGB images.
    """
    w, h = image.size
    target_w, target_h = size
    target_ratio = target_w / target_h
    image_ratio = w / h

    if image_ratio > target_ratio:
        # Image is wider — crop width
        new_w = int(h * target_ratio)
        new_h = h
    else:
        # Image is taller — crop height
        new_w = w
        new_h = int(w / target_ratio)

    # Center crop
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    image = image.crop((left, top, left + new_w, top + new_h))
    # Resize to exact target
    image = image.resize(size, resample)
    return image


def resize_and_padding(image, size):
    """
    Resize a PIL image to fit within `size` (width, height) with letterbox padding.

    Strategy:
      1. Scale the image so it fits entirely within the target dimensions.
      2. Pad the remaining space with white (255, 255, 255).
      3. The image is centered within the padded frame.

    This keeps the full garment visible without cropping.
    """
    w, h = image.size
    target_w, target_h = size
    target_ratio = target_w / target_h
    image_ratio = w / h

    if image_ratio > target_ratio:
        # Image is wider — fit to width
        new_w = target_w
        new_h = int(target_w / image_ratio)
    else:
        # Image is taller — fit to height
        new_h = target_h
        new_w = int(target_h * image_ratio)

    image = image.resize((new_w, new_h), Image.LANCZOS)
    # Center on white canvas
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding


def scan_files_in_dir(directory, postfix: Set[str] = None, progress_bar: tqdm = None) -> list:
    file_list = []
    progress_bar = tqdm(total=0, desc=f"Scanning", ncols=100) if progress_bar is None else progress_bar
    for entry in os.scandir(directory):
        if entry.is_file():
            if postfix is None or os.path.splitext(entry.path)[1] in postfix:
                file_list.append(entry)
                progress_bar.total += 1
                progress_bar.update(1)
        elif entry.is_dir():
            file_list += scan_files_in_dir(entry.path, postfix=postfix, progress_bar=progress_bar)
    return file_list

if __name__ == "__main__":
    ...