"""
DensePose predictor — standalone copy.

Given a person image (PIL.Image or file path), runs Detectron2 DensePose
and returns a grayscale PIL.Image with body-part segment indices.

Checkpoint download:
  Model  : https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
  Config : https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml
           (also needs Base-DensePose-RCNN-FPN.yaml in the same folder)

Place both files under the directory you pass as `model_path`.
"""

import glob
import os
from random import randint
import shutil
import time

import cv2
import numpy as np
import torch
from PIL import Image
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import create_extractor, CompoundExtractor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor


class DensePosePredictor:
    """
    Wraps Detectron2 DensePose to produce a grayscale body-part segmentation map.

    Args:
        model_path (str): Directory containing:
            - densepose_rcnn_R_50_FPN_s1x.yaml
            - Base-DensePose-RCNN-FPN.yaml
            - model_final_162be9.pkl
        device (str): 'cuda' or 'cpu'
    """

    def __init__(self, model_path: str = "./Models/DensePose", device: str = "cuda"):
        self.device = device
        self.config_path = os.path.join(model_path, "densepose_rcnn_R_50_FPN_s1x.yaml")
        self.model_path = os.path.join(model_path, "model_final_162be9.pkl")
        self.visualizations = ["dp_segm"]
        self.VISUALIZERS = {"dp_segm": DensePoseResultsFineSegmentationVisualizer}
        self.min_score = 0.8

        self.cfg = self._setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_config(self):
        opts = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(self.min_score)]
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.freeze()
        return cfg

    @staticmethod
    def _get_input_file_list(input_spec: str):
        if os.path.isdir(input_spec):
            return [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            return [input_spec]
        else:
            return glob.glob(input_spec)

    def _create_context(self, cfg, output_path: str) -> dict:
        visualizers, extractors = [], []
        for vis_spec in self.visualizations:
            vis = self.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=None,
                texture_atlases_dict=None,
                alpha=1.0,
            )
            visualizers.append(vis)
            extractors.append(create_extractor(vis))
        return {
            "extractor": CompoundExtractor(extractors),
            "visualizer": CompoundVisualizer(visualizers),
            "out_fname": output_path,
            "entry_idx": 0,
        }

    def _execute_on_outputs(self, context: dict, entry: dict, outputs) -> None:
        data = context["extractor"](outputs)
        H, W, _ = entry["image"].shape
        result = np.zeros((H, W), dtype=np.uint8)

        data, box = data[0]
        x, y, w, h = [int(_) for _ in box[0].cpu().numpy()]
        i_array = data[0].labels[None].cpu().numpy()[0]
        result[y : y + h, x : x + w] = i_array

        Image.fromarray(result).save(context["out_fname"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image_or_path, resize: int = 512) -> Image.Image:
        """
        Run DensePose on a single image.

        Args:
            image_or_path: File path (str) or PIL.Image.Image.
            resize (int): Resize the longest side to this value before inference.

        Returns:
            PIL.Image.Image: Grayscale image where each pixel value is a
            DensePose body-part index (0 = background).
        """
        tmp_dir = "./mask_generation_tmp/"
        os.makedirs(tmp_dir, exist_ok=True)

        image_path = os.path.join(
            tmp_dir,
            f"{int(time.time())}-{self.device}-{randint(0, 100000)}.png",
        )

        if isinstance(image_or_path, str):
            assert image_or_path.split(".")[-1].lower() in (
                "jpg",
                "jpeg",
                "png",
            ), "Only jpg/jpeg/png images are supported."
            shutil.copy(image_or_path, image_path)
        elif isinstance(image_or_path, Image.Image):
            image_or_path.save(image_path)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise TypeError("image_or_path must be a file path (str) or PIL.Image.Image")

        output_path = image_path.replace(".png", "_dense.png")
        orig_w, orig_h = Image.open(image_path).size

        file_list = self._get_input_file_list(image_path)
        assert len(file_list), "No input images found!"

        context = self._create_context(self.cfg, output_path)
        for file_name in file_list:
            img = read_image(file_name, format="BGR")
            if (_ := max(img.shape)) > resize:
                scale = resize / _
                img = cv2.resize(
                    img,
                    (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                )
            with torch.no_grad():
                outputs = self.predictor(img)["instances"]
                try:
                    self._execute_on_outputs(
                        context, {"file_name": file_name, "image": img}, outputs
                    )
                except Exception:
                    # No person detected — save a blank mask
                    Image.new("L", (1, 1)).save(output_path)

        dense_gray = Image.open(output_path).convert("L")
        dense_gray = dense_gray.resize((orig_w, orig_h), Image.NEAREST)

        # Clean up temp files
        os.remove(image_path)
        os.remove(output_path)

        return dense_gray
