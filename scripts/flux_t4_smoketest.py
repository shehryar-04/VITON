#!/usr/bin/env python
"""
Flux try-on smoke test for Tesla T4 (Colab).
=============================================

Validates that FLUX.1-Fill-dev loads and runs a single try-on on a 16GB T4
*before* wiring it into the full worker. It mirrors the worker's loader
(NF4 4-bit transformer + fp32 VAE) so what passes here is what the worker uses.

It checks the three things that can't be verified without the actual GPU/weights:
  1. Memory fit  — prints peak VRAM after load and after generation.
  2. dtype/VAE   — flags black/NaN output (the classic Flux-fp16-VAE failure).
  3. Versions    — surfaces diffusers / bitsandbytes import errors early.

Usage (Colab cell or shell)::

    huggingface-cli login          # gated FLUX.1-Fill-dev access
    pip install bitsandbytes accelerate diffusers

    python scripts/flux_t4_smoketest.py \
        --person path/to/person.jpg \
        --cloth  path/to/garment.jpg \
        --out    result.png

If --person/--cloth are omitted, synthetic test images are generated so you can
still validate load + memory + decode without real inputs.

Run from the repo root so `model` and `worker` are importable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from PIL import Image

# Make the repo root importable no matter how the script is launched
# (`python scripts/flux_t4_smoketest.py` only puts scripts/ on sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _log(msg: str) -> None:
    print(f"[smoketest] {msg}", flush=True)


def _vram(torch, label: str) -> None:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        _log(f"VRAM {label}: peak_allocated={peak:.2f} GB, reserved={reserved:.2f} GB")


def _make_dummy(width: int, height: int) -> tuple[Image.Image, Image.Image]:
    """Synthetic person + textured garment so the run works without real inputs."""
    import numpy as np

    person = Image.new("RGB", (width, height), (180, 170, 160))
    # High-frequency checkerboard so texture loss is visually obvious.
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    tile = 16
    for y in range(height):
        for x in range(width):
            arr[y, x] = (230, 60, 60) if ((x // tile) + (y // tile)) % 2 else (60, 60, 230)
    garment = Image.fromarray(arr)
    return person, garment


def main() -> int:
    parser = argparse.ArgumentParser(description="Flux try-on T4 smoke test")
    parser.add_argument("--person", default=None, help="person image path")
    parser.add_argument("--cloth", default=None, help="garment image path")
    parser.add_argument("--mask", default=None, help="mask path (white=inpaint); auto if omitted")
    parser.add_argument("--cloth-type", default="upper",
                        choices=["upper", "lower", "overall", "inner", "outer"],
                        help="garment type for AutoMasker (default: upper)")
    parser.add_argument("--out", default="flux_smoketest_result.png", help="output image path")
    parser.add_argument("--base", default=os.environ.get("FLUX_BASE_CKPT", "black-forest-labs/FLUX.1-Fill-dev"),
                        help="base model for VAE + scheduler")
    parser.add_argument("--transformer", default=os.environ.get("FLUX_TRANSFORMER_CKPT", ""),
                        help="fine-tuned try-on transformer (e.g. xiaozaa/catvton-flux-alpha). "
                             "Without it the base Fill model ignores the garment.")
    parser.add_argument("--transformer-subfolder", default=os.environ.get("FLUX_TRANSFORMER_SUBFOLDER", ""),
                        help="subfolder of the transformer repo ('' = root)")
    parser.add_argument("--lora", default=os.environ.get("FLUX_LORA_PATH", ""), help="optional try-on LoRA path")
    parser.add_argument("--height", type=int, default=int(os.environ.get("FLUX_HEIGHT", "1024")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("FLUX_WIDTH", "768")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("FLUX_NUM_INFERENCE_STEPS", "30")))
    parser.add_argument("--guidance", type=float, default=float(os.environ.get("FLUX_GUIDANCE_SCALE", "30")))
    parser.add_argument("--no-quant", action="store_true", help="disable NF4 4-bit (needs ~24GB)")
    parser.add_argument("--vae-compute-dtype", action="store_true", help="run VAE in compute dtype instead of fp32")
    args = parser.parse_args()

    # Belt-and-suspenders validation for programmatic invocation
    VALID_CLOTH_TYPES = {"upper", "lower", "overall", "inner", "outer"}
    if args.cloth_type not in VALID_CLOTH_TYPES:
        _log(f"ERROR: invalid --cloth-type '{args.cloth_type}'. Must be one of: {', '.join(sorted(VALID_CLOTH_TYPES))}")
        return 1

    try:
        import torch
    except ImportError:
        _log("ERROR: torch not installed.")
        return 1

    if not torch.cuda.is_available():
        _log("ERROR: no CUDA device visible. Run on a GPU runtime (Colab: Runtime > Change runtime type > T4).")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    native_bf16 = major >= 8  # Ampere+ ; T4 (7.5) has no native bf16
    _log(
        f"GPU: {gpu_name} | sm_{major}{minor} | native_bf16={native_bf16} | "
        f"compute_dtype={'bf16' if native_bf16 else 'fp16'}"
    )
    if "T4" not in gpu_name and native_bf16:
        _log("note: this script is tuned for T4 (fp16); bf16 GPU detected — that's fine, just not the target.")

    # Build the pipeline via the SAME code path the worker uses.
    from worker.inference_engine import InferenceEngine
    from worker.models import InferenceConfig

    config = InferenceConfig(
        pipeline_type="flux",
        device="cuda:0",
        batch_size=1,
        flash_attention=False,
        vae_tiling=True,
        vae_slicing=False,
        vae_tiling_resolution=1024,
        flux_base_ckpt=args.base,
        flux_transformer_ckpt=args.transformer,
        flux_transformer_subfolder=args.transformer_subfolder,
        flux_lora_path=args.lora,
        flux_quantize_4bit=not args.no_quant,
        flux_vae_fp32=not args.vae_compute_dtype,
        flux_cpu_offload="none",
        flux_height=args.height,
        flux_width=args.width,
        flux_num_inference_steps=args.steps,
        flux_guidance_scale=args.guidance,
    )

    # Minimal shim so we reuse _build_flux_pipeline without a real config module.
    class _Cfg:
        FLUX_BASE_CKPT = args.base
        FLUX_CKPT = ""

    engine = object.__new__(InferenceEngine)
    engine.config = config

    torch.cuda.reset_peak_memory_stats()
    if args.transformer:
        _log(f"Try-on transformer: '{args.transformer}' (subfolder='{args.transformer_subfolder or '(root)'}')")
    else:
        _log("Try-on transformer: NONE — using base Fill transformer. "
             "This is GENERIC INPAINTING; the garment WILL be ignored (expect a flat/gray fill).")
    _log(f"Loading Flux pipeline from '{args.base}' (4bit={not args.no_quant}, vae_fp32={not args.vae_compute_dtype}) ...")
    t0 = time.time()
    from model.flux.pipeline_flux_tryon import FluxTryOnPipeline
    try:
        pipeline = engine._build_flux_pipeline(config, _Cfg, FluxTryOnPipeline, torch)
        if config.vae_tiling:
            pipeline.enable_vae_tiling()
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR during load: {type(exc).__name__}: {exc}")
        _log("Common causes: not logged in (gated repo), bitsandbytes missing, or diffusers version mismatch.")
        return 1
    _log(f"Loaded in {time.time() - t0:.1f}s")
    _vram(torch, "after load")

    # Inputs
    if args.person and args.cloth:
        person = Image.open(args.person).convert("RGB").resize((args.width, args.height))
        cloth = Image.open(args.cloth).convert("RGB").resize((args.width, args.height))
        _log("Using provided person/cloth images.")
    else:
        person, cloth = _make_dummy(args.width, args.height)
        _log("Using synthetic person/cloth images (no --person/--cloth given).")
    # --- Mask resolution flow ---
    if args.mask:
        # Manual mask override
        if not os.path.isfile(args.mask):
            _log(f"ERROR: mask path not found or not readable: {args.mask}")
            return 1
        try:
            mask = Image.open(args.mask).convert("L").resize((args.width, args.height))
            _log(f"Using manual mask from: {args.mask}")
        except Exception as exc:
            _log(f"ERROR: could not open mask file '{args.mask}': {exc}")
            return 1
    else:
        # Try AutoMasker
        auto_masker = None
        try:
            from model.cloth_masker import AutoMasker
            import worker.config as cfg
            auto_masker = AutoMasker(
                densepose_ckpt=cfg.DENSEPOSE_CKPT or "./Models/DensePose",
                schp_ckpt=cfg.SCHP_CKPT or "./Models/SCHP",
                device="cuda:0",
            )
            _log("AutoMasker loaded for mask generation.")
        except Exception as exc:
            _log(f"AutoMasker unavailable ({exc}) — falling back to blank mask.")

        if auto_masker is not None:
            result = auto_masker(person, mask_type=args.cloth_type)
            mask = result["mask"].resize((args.width, args.height), Image.NEAREST)
            _log(f"Generated AutoMasker mask with cloth_type='{args.cloth_type}'")
        else:
            _log("WARNING: No mask source available — using blank mask (all white, full regeneration).")
            mask = Image.new("L", (args.width, args.height), 255)

    _log(f"Running inference: {args.steps} steps, guidance={args.guidance}, {args.width}x{args.height} ...")
    t0 = time.time()
    try:
        out = pipeline(
            image=person,
            condition_image=cloth,
            mask_image=mask,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR during inference: {type(exc).__name__}: {exc}")
        return 1
    dt = time.time() - t0
    _log(f"Inference done in {dt:.1f}s ({dt / args.steps:.2f}s/step)")
    _vram(torch, "after inference")

    image = out.images[0] if hasattr(out, "images") else out[0]

    # Black/NaN check — the classic Flux-fp16-VAE failure mode.
    import numpy as np
    a = np.asarray(image.convert("RGB")).astype(np.float32)
    mean, std = a.mean(), a.std()
    _log(f"Output stats: mean={mean:.2f}, std={std:.2f}")
    if mean < 2.0 or std < 1.0:
        _log("WARNING: output looks black/flat — likely VAE precision issue.")
        _log("Fix: ensure VAE runs in fp32 (default). Do NOT pass --vae-compute-dtype.")

    image.save(args.out)
    _log(f"Saved result to {args.out}")
    _log("SMOKE TEST PASSED" if (mean >= 2.0 and std >= 1.0) else "SMOKE TEST COMPLETED WITH WARNINGS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
