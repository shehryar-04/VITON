# Mask Generation Package
# ========================
# Standalone cloth-agnostic mask generation pipeline.
#
# Quick start
# -----------
#   from mask_generation import MaskGenerator
#
#   gen = MaskGenerator(
#       densepose_ckpt='./Models/DensePose',
#       schp_ckpt='./Models/SCHP',
#       device='cuda',
#   )
#   result = gen('person.jpg', mask_type='upper')
#
#   result['mask'].save('mask.png')          # grayscale mask (0/255)
#   result['densepose'].save('dp.png')       # DensePose body-part map
#   result['schp_lip'].save('lip.png')       # LIP clothing parse
#   result['schp_atr'].save('atr.png')       # ATR clothing parse
#
# Supported mask_type values
# --------------------------
#   'upper'   – shirt / top / coat / dress  (torso + arms)
#   'lower'   – pants / skirt               (thighs + legs)
#   'overall' – full outfit                 (torso + arms + legs)
#   'inner'   – inner layer only            (e.g. shirt under jacket)
#   'outer'   – outer layer only            (e.g. jacket / coat)

from mask_generation.masker import (
    MaskGenerator,
    compute_cloth_agnostic_mask,
    vis_mask,
    part_mask_of,
    hull_mask,
    DENSE_INDEX_MAP,
    ATR_MAPPING,
    LIP_MAPPING,
    VALID_MASK_TYPES,
)
from mask_generation.densepose_predictor import DensePosePredictor
from mask_generation.schp_predictor import SCHPPredictor

__all__ = [
    "MaskGenerator",
    "compute_cloth_agnostic_mask",
    "vis_mask",
    "part_mask_of",
    "hull_mask",
    "DensePosePredictor",
    "SCHPPredictor",
    "DENSE_INDEX_MAP",
    "ATR_MAPPING",
    "LIP_MAPPING",
    "VALID_MASK_TYPES",
]
