"""
CLI entry point for the standalone mask generation pipeline.

Models are downloaded automatically from HuggingFace (zhengchong/CatVTON)
the first time you run this — same as the Ngrok-GPU-Endpoint notebook.

Usage
-----
    # Simplest — auto-downloads models from HuggingFace:
    python mask_generation/generate_mask.py --image mask_generation/kurta.jpg --mask_type upper

    # With explicit local model paths (if already downloaded):
    python mask_generation/generate_mask.py \\
        --image      mask_generation/kurta.jpg \\
        --mask_type  upper \\
        --densepose  /path/to/CatVTON/DensePose \\
        --schp       /path/to/CatVTON/SCHP \\
        --out_dir    ./output

Outputs saved to --out_dir:
    mask.png       – binary cloth-agnostic mask  (0 = keep, 255 = replace)
    densepose.png  – DensePose body-part map
    schp_lip.png   – LIP clothing parse
    schp_atr.png   – ATR clothing parse
    overlay.png    – original image with masked region blacked out (visual check)
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

# Allow running from the virtual-Try-On root:
#   python mask_generation/generate_mask.py ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_generation import MaskGenerator, vis_mask


def _resolve_model_paths(densepose_arg: str, schp_arg: str) -> tuple[str, str]:
    """
    If the user didn't supply explicit paths (or the supplied paths don't exist),
    download the zhengchong/CatVTON repo from HuggingFace — exactly as the
    Ngrok-GPU-Endpoint notebook does — and return the DensePose / SCHP sub-dirs.
    """
    # If both paths were given and exist, use them directly
    if (
        densepose_arg
        and schp_arg
        and os.path.isdir(densepose_arg)
        and os.path.isdir(schp_arg)
    ):
        return densepose_arg, schp_arg

    # Otherwise fall back to snapshot_download (same as the notebook)
    print("Model paths not found locally — downloading zhengchong/CatVTON from HuggingFace ...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    repo_folder = snapshot_download(repo_id="zhengchong/CatVTON")
    print(f"Downloaded to: {repo_folder}")

    densepose_path = os.path.join(repo_folder, "DensePose")
    schp_path      = os.path.join(repo_folder, "SCHP")

    if not os.path.isdir(densepose_path):
        print(f"[ERROR] DensePose folder not found inside downloaded repo: {densepose_path}")
        sys.exit(1)
    if not os.path.isdir(schp_path):
        print(f"[ERROR] SCHP folder not found inside downloaded repo: {schp_path}")
        sys.exit(1)

    return densepose_path, schp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cloth-agnostic mask for a person image."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input person image (jpg or png).",
    )
    parser.add_argument(
        "--mask_type", default="upper",
        choices=["upper", "lower", "overall", "inner", "outer"],
        help="Which garment region to mask (default: upper).",
    )
    parser.add_argument(
        "--densepose", default="",
        help=(
            "Path to the DensePose model directory. "
            "If omitted (or path doesn't exist), models are auto-downloaded "
            "from HuggingFace (zhengchong/CatVTON)."
        ),
    )
    parser.add_argument(
        "--schp", default="",
        help=(
            "Path to the SCHP model directory (must contain both .pth files). "
            "If omitted (or path doesn't exist), models are auto-downloaded "
            "from HuggingFace (zhengchong/CatVTON)."
        ),
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to run inference on: 'cuda' or 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--out_dir", default="./mask_output",
        help="Directory to save output images (default: ./mask_output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate input image
    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve model paths — download from HF if not provided / not found locally
    densepose_path, schp_path = _resolve_model_paths(args.densepose, args.schp)

    print(f"DensePose : {densepose_path}")
    print(f"SCHP      : {schp_path}")
    print(f"Device    : {args.device}")
    print(f"Loading models ...")

    generator = MaskGenerator(
        densepose_ckpt=densepose_path,
        schp_ckpt=schp_path,
        device=args.device,
    )

    print(f"Processing: {args.image}  (mask_type={args.mask_type})")
    person_image = Image.open(args.image).convert("RGB")
    result = generator(person_image, mask_type=args.mask_type)

    # Save all outputs
    mask_path      = os.path.join(args.out_dir, "mask.png")
    densepose_path = os.path.join(args.out_dir, "densepose.png")
    lip_path       = os.path.join(args.out_dir, "schp_lip.png")
    atr_path       = os.path.join(args.out_dir, "schp_atr.png")
    overlay_path   = os.path.join(args.out_dir, "overlay.png")

    result["mask"].save(mask_path)
    result["densepose"].save(densepose_path)
    result["schp_lip"].save(lip_path)
    result["schp_atr"].save(atr_path)
    vis_mask(person_image, result["mask"]).save(overlay_path)

    print(f"\nOutputs saved to: {args.out_dir}")
    print(f"  mask.png      → {mask_path}")
    print(f"  densepose.png → {densepose_path}")
    print(f"  schp_lip.png  → {lip_path}")
    print(f"  schp_atr.png  → {atr_path}")
    print(f"  overlay.png   → {overlay_path}")


if __name__ == "__main__":
    main()
