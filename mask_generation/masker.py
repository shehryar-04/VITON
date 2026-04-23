"""
Cloth-agnostic mask generator — standalone copy.

Combines DensePose body-part segmentation and SCHP clothing parsing
to produce a binary mask of the region to be replaced during try-on.

Usage
-----
    from mask_generation.masker import MaskGenerator

    gen = MaskGenerator(
        densepose_ckpt='./Models/DensePose',
        schp_ckpt='./Models/SCHP',
        device='cuda',
    )

    result = gen(image, mask_type='upper')
    # result['mask']       → PIL.Image  (grayscale, 0/255)
    # result['densepose']  → PIL.Image  (body-part index map)
    # result['schp_lip']   → PIL.Image  (LIP clothing parse)
    # result['schp_atr']   → PIL.Image  (ATR clothing parse)

Supported mask_type values
--------------------------
    'upper'   – shirt, top, coat, dress (torso + arms)
    'lower'   – pants, skirt (thighs + legs)
    'overall' – full outfit (torso + arms + legs)
    'inner'   – inner layer only (e.g. shirt under jacket)
    'outer'   – outer layer only (e.g. jacket / coat)
"""

from __future__ import annotations

import os
from typing import Union

import cv2
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from mask_generation.densepose_predictor import DensePosePredictor
from mask_generation.schp_predictor import SCHPPredictor

# ---------------------------------------------------------------------------
# Label-index maps
# ---------------------------------------------------------------------------

DENSE_INDEX_MAP = {
    "background":    [0],
    "torso":         [1, 2],
    "right hand":    [3],
    "left hand":     [4],
    "right foot":    [5],
    "left foot":     [6],
    "right thigh":   [7, 9],
    "left thigh":    [8, 10],
    "right leg":     [11, 13],
    "left leg":      [12, 14],
    "left big arm":  [15, 17],
    "right big arm": [16, 18],
    "left forearm":  [19, 21],
    "right forearm": [20, 22],
    "face":          [23, 24],
    "thighs":        [7, 8, 9, 10],
    "legs":          [11, 12, 13, 14],
    "hands":         [3, 4],
    "feet":          [5, 6],
    "big arms":      [15, 16, 17, 18],
    "forearms":      [19, 20, 21, 22],
}

ATR_MAPPING = {
    "Background": 0, "Hat": 1, "Hair": 2, "Sunglasses": 3,
    "Upper-clothes": 4, "Skirt": 5, "Pants": 6, "Dress": 7,
    "Belt": 8, "Left-shoe": 9, "Right-shoe": 10, "Face": 11,
    "Left-leg": 12, "Right-leg": 13, "Left-arm": 14, "Right-arm": 15,
    "Bag": 16, "Scarf": 17,
}

LIP_MAPPING = {
    "Background": 0, "Hat": 1, "Hair": 2, "Glove": 3,
    "Sunglasses": 4, "Upper-clothes": 5, "Dress": 6, "Coat": 7,
    "Socks": 8, "Pants": 9, "Jumpsuits": 10, "Scarf": 11,
    "Skirt": 12, "Face": 13, "Left-arm": 14, "Right-arm": 15,
    "Left-leg": 16, "Right-leg": 17, "Left-shoe": 18, "Right-shoe": 19,
}

# ---------------------------------------------------------------------------
# Per-mask-type region definitions
# ---------------------------------------------------------------------------

PROTECT_BODY_PARTS = {
    "upper":   ["Left-leg", "Right-leg"],
    "lower":   ["Right-arm", "Left-arm", "Face"],
    "overall": [],
    "inner":   ["Left-leg", "Right-leg"],
    "outer":   ["Left-leg", "Right-leg"],
}

PROTECT_CLOTH_PARTS = {
    "upper":   {"ATR": ["Skirt", "Pants"],                          "LIP": ["Skirt", "Pants"]},
    "lower":   {"ATR": ["Upper-clothes"],                           "LIP": ["Upper-clothes", "Coat"]},
    "overall": {"ATR": [],                                          "LIP": []},
    "inner":   {"ATR": ["Dress", "Coat", "Skirt", "Pants"],         "LIP": ["Dress", "Coat", "Skirt", "Pants", "Jumpsuits"]},
    "outer":   {"ATR": ["Dress", "Pants", "Skirt"],                 "LIP": ["Upper-clothes", "Dress", "Pants", "Skirt", "Jumpsuits"]},
}

MASK_CLOTH_PARTS = {
    "upper":   ["Upper-clothes", "Coat", "Dress", "Jumpsuits"],
    "lower":   ["Pants", "Skirt", "Dress", "Jumpsuits"],
    "overall": ["Upper-clothes", "Dress", "Pants", "Skirt", "Coat", "Jumpsuits"],
    "inner":   ["Upper-clothes"],
    "outer":   ["Coat"],
}

MASK_DENSE_PARTS = {
    "upper":   ["torso", "big arms", "forearms"],
    "lower":   ["thighs", "legs"],
    "overall": ["torso", "thighs", "legs", "big arms", "forearms"],
    "inner":   ["torso"],
    "outer":   ["torso", "big arms", "forearms"],
}

VALID_MASK_TYPES = ("upper", "lower", "overall", "inner", "outer")

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def part_mask_of(
    part: Union[str, list],
    parse: np.ndarray,
    mapping: dict,
) -> np.ndarray:
    """Return a binary numpy array selecting the given body/cloth part(s)."""
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for p in part:
        if p not in mapping:
            continue
        indices = mapping[p] if isinstance(mapping[p], list) else [mapping[p]]
        for idx in indices:
            mask += (parse == idx)
    return mask


def hull_mask(mask_area: np.ndarray) -> np.ndarray:
    """Fill convex hulls of all contours in a binary mask."""
    _, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        result = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | result
    return result


def vis_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Overlay mask on image for visualisation (masked region → black)."""
    img_arr = np.array(image).astype(np.uint8)
    msk_arr = np.array(mask).astype(np.uint8)
    msk_arr = np.where(msk_arr > 127, 255, 0)
    msk_3ch = np.repeat(msk_arr[:, :, np.newaxis], 3, axis=-1) / 255.0
    return Image.fromarray((img_arr * (1 - msk_3ch)).astype(np.uint8))


# ---------------------------------------------------------------------------
# Core mask computation (static, no model dependencies)
# ---------------------------------------------------------------------------

def compute_cloth_agnostic_mask(
    densepose_mask: Image.Image,
    schp_lip_mask: Image.Image,
    schp_atr_mask: Image.Image,
    part: str = "overall",
) -> Image.Image:
    """
    Compute a cloth-agnostic binary mask from pre-computed segmentation maps.

    Args:
        densepose_mask: Grayscale PIL image with DensePose body-part indices.
        schp_lip_mask:  Palette PIL image with LIP clothing-parse labels.
        schp_atr_mask:  Palette PIL image with ATR clothing-parse labels.
        part:           One of 'upper', 'lower', 'overall', 'inner', 'outer'.

    Returns:
        PIL.Image grayscale mask (0 = keep, 255 = replace).
    """
    assert part in VALID_MASK_TYPES, (
        f"part must be one of {VALID_MASK_TYPES}, got {part!r}"
    )

    w, h = densepose_mask.size

    # Kernel sizes derived from image dimensions
    dilate_k = max(w, h) // 250
    dilate_k = dilate_k if dilate_k % 2 == 1 else dilate_k + 1
    dilate_kernel = np.ones((dilate_k, dilate_k), np.uint8)

    blur_k = max(w, h) // 25
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1

    dp  = np.array(densepose_mask)
    lip = np.array(schp_lip_mask)
    atr = np.array(schp_atr_mask)

    # ---- Strong protect: hands, feet, face --------------------------------
    hands_area = part_mask_of(["hands", "feet"], dp, DENSE_INDEX_MAP)
    hands_area = cv2.dilate(hands_area, dilate_kernel, iterations=1)
    hands_area = hands_area & (
        part_mask_of(["Left-arm", "Right-arm", "Left-leg", "Right-leg"], atr, ATR_MAPPING)
        | part_mask_of(["Left-arm", "Right-arm", "Left-leg", "Right-leg"], lip, LIP_MAPPING)
    )
    face_area = part_mask_of("Face", lip, LIP_MAPPING)
    strong_protect = hands_area | face_area

    # ---- Weak protect: hair, irrelevant clothes, body parts ---------------
    body_protect = (
        part_mask_of(PROTECT_BODY_PARTS[part], lip, LIP_MAPPING)
        | part_mask_of(PROTECT_BODY_PARTS[part], atr, ATR_MAPPING)
    )
    hair_protect = (
        part_mask_of(["Hair"], lip, LIP_MAPPING)
        | part_mask_of(["Hair"], atr, ATR_MAPPING)
    )
    cloth_protect = (
        part_mask_of(PROTECT_CLOTH_PARTS[part]["LIP"], lip, LIP_MAPPING)
        | part_mask_of(PROTECT_CLOTH_PARTS[part]["ATR"], atr, ATR_MAPPING)
    )
    accessory_parts = ["Hat", "Glove", "Sunglasses", "Bag", "Left-shoe", "Right-shoe", "Scarf", "Socks"]
    accessory_protect = (
        part_mask_of(accessory_parts, lip, LIP_MAPPING)
        | part_mask_of(accessory_parts, atr, ATR_MAPPING)
    )
    weak_protect = body_protect | cloth_protect | hair_protect | strong_protect | accessory_protect

    # ---- Mask area --------------------------------------------------------
    strong_mask = (
        part_mask_of(MASK_CLOTH_PARTS[part], lip, LIP_MAPPING)
        | part_mask_of(MASK_CLOTH_PARTS[part], atr, ATR_MAPPING)
    )
    background = (
        part_mask_of(["Background"], lip, LIP_MAPPING)
        & part_mask_of(["Background"], atr, ATR_MAPPING)
    )
    dense_mask = part_mask_of(MASK_DENSE_PARTS[part], dp, DENSE_INDEX_MAP)
    # Downscale → dilate → upscale (cheap morphological expansion)
    dense_mask = cv2.resize(dense_mask.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    dense_mask = cv2.dilate(dense_mask, dilate_kernel, iterations=2)
    dense_mask = cv2.resize(dense_mask.astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    mask_area = (np.ones_like(dp) & (~weak_protect) & (~background)) | dense_mask
    mask_area = hull_mask(mask_area * 255) // 255          # convex hull expansion
    mask_area = mask_area & (~weak_protect)
    mask_area = cv2.GaussianBlur(mask_area * 255, (blur_k, blur_k), 0)
    mask_area = np.where(mask_area >= 25, 1, 0).astype(np.uint8)
    mask_area = (mask_area | strong_mask) & (~strong_protect)
    mask_area = cv2.dilate(mask_area, dilate_kernel, iterations=1)

    return Image.fromarray((mask_area * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class MaskGenerator:
    """
    End-to-end cloth-agnostic mask generator.

    Loads DensePose and two SCHP models (LIP + ATR) once, then accepts
    any number of images via __call__.

    Args:
        densepose_ckpt (str): Path to the DensePose model directory.
        schp_ckpt (str):      Path to the SCHP model directory containing
                              both .pth checkpoint files.
        device (str):         'cuda' or 'cpu'

    Example::

        gen = MaskGenerator('./Models/DensePose', './Models/SCHP', device='cuda')
        result = gen('person.jpg', mask_type='upper')
        result['mask'].save('mask.png')
    """

    def __init__(
        self,
        densepose_ckpt: str = "./Models/DensePose",
        schp_ckpt: str = "./Models/SCHP",
        device: str = "cuda",
    ) -> None:
        np.random.seed(0)
        torch.manual_seed(0)
        if device.startswith("cuda"):
            torch.cuda.manual_seed(0)

        self.densepose = DensePosePredictor(model_path=densepose_ckpt, device=device)
        self.schp_atr = SCHPPredictor(
            ckpt_path=os.path.join(schp_ckpt, "exp-schp-201908301523-atr.pth"),
            device=device,
        )
        self.schp_lip = SCHPPredictor(
            ckpt_path=os.path.join(schp_ckpt, "exp-schp-201908261155-lip.pth"),
            device=device,
        )
        # Optional: VAE-compatible binarising processor (mirrors original code)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    # ------------------------------------------------------------------
    # Preprocessing helpers (exposed for external use)
    # ------------------------------------------------------------------

    def run_densepose(self, image_or_path, resize: int = 1024) -> Image.Image:
        """Return the DensePose body-part map for an image."""
        return self.densepose(image_or_path, resize=resize)

    def run_schp_lip(self, image_or_path) -> Image.Image:
        """Return the LIP clothing-parse map for an image."""
        return self.schp_lip(image_or_path)

    def run_schp_atr(self, image_or_path) -> Image.Image:
        """Return the ATR clothing-parse map for an image."""
        return self.schp_atr(image_or_path)

    def preprocess_image(self, image_or_path) -> dict:
        """
        Run all three segmentation models and return their outputs.

        Returns:
            dict with keys: 'densepose', 'schp_atr', 'schp_lip'
        """
        return {
            "densepose": self.densepose(image_or_path, resize=1024),
            "schp_atr":  self.schp_atr(image_or_path),
            "schp_lip":  self.schp_lip(image_or_path),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: Union[str, Image.Image],
        mask_type: str = "upper",
    ) -> dict:
        """
        Generate a cloth-agnostic mask for the given person image.

        Args:
            image:     File path (str) or PIL.Image.Image of the person.
            mask_type: Region to mask. One of:
                       'upper', 'lower', 'overall', 'inner', 'outer'

        Returns:
            dict:
                'mask'      → PIL.Image grayscale (0 = keep, 255 = replace)
                'densepose' → PIL.Image DensePose body-part map
                'schp_lip'  → PIL.Image LIP clothing parse
                'schp_atr'  → PIL.Image ATR clothing parse
        """
        assert mask_type in VALID_MASK_TYPES, (
            f"mask_type must be one of {VALID_MASK_TYPES}, got {mask_type!r}"
        )

        preprocessed = self.preprocess_image(image)
        mask = compute_cloth_agnostic_mask(
            densepose_mask=preprocessed["densepose"],
            schp_lip_mask=preprocessed["schp_lip"],
            schp_atr_mask=preprocessed["schp_atr"],
            part=mask_type,
        )

        return {
            "mask":      mask,
            "densepose": preprocessed["densepose"],
            "schp_lip":  preprocessed["schp_lip"],
            "schp_atr":  preprocessed["schp_atr"],
        }
