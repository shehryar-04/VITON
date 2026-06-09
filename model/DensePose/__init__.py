
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


class DensePose:
    """
    DensePose used in this project is from Detectron2 (https://github.com/facebookresearch/detectron2).
    These codes are modified from https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose.
    The checkpoint is downloaded from https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo.

    We use the model R_50_FPN_s1x with id 165712039, but other models should also work.
    The config file is downloaded from https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose/configs.
    Noted that the config file should match the model checkpoint and Base-DensePose-RCNN-FPN.yaml is also needed.
    """

    def __init__(self, model_path="./checkpoints/densepose_", device="cuda"):
        self.device = device
        self.config_path = os.path.join(model_path, 'densepose_rcnn_R_50_FPN_s1x.yaml')
        self.model_path = os.path.join(model_path, 'model_final_162be9.pkl')
        self.visualizations = ["dp_segm"]
        self.VISUALIZERS = {"dp_segm": DensePoseResultsFineSegmentationVisualizer}
        self.min_score = 0.8

        self.cfg = self.setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.to(self.device)

    def setup_config(self):
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
            file_list = [os.path.join(input_spec, fname) for fname in os.listdir(input_spec)
                         if os.path.isfile(os.path.join(input_spec, fname))]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list

    def create_context(self, cfg, output_path):
        vis_specs = self.visualizations
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = texture_atlases_dict = None
            vis = self.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
                alpha=1.0
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": output_path,
            "entry_idx": 0,
        }
        return context

    def execute_on_outputs(self, context, entry, outputs):
        extractor = context["extractor"]

        data = extractor(outputs)

        H, W, _ = entry["image"].shape
        result = np.zeros((H, W), dtype=np.uint8)

        data, box = data[0]
        x, y, w, h = [int(_) for _ in box[0].cpu().numpy()]
        i_array = data[0].labels[None].cpu().numpy()[0]
        result[y:y + h, x:x + w] = i_array
        result = Image.fromarray(result)
        result.save(context["out_fname"])

    def execute_on_outputs_iuv(self, entry, outputs):
        """
        Extract raw IUV maps from a DensePose prediction.

        Returns:
            np.ndarray of shape (H, W, 3) float32.
            Channel 0 = I  — raw body-part index as float32, integer values in [0, 24].
                             Background pixels = 0.
            Channel 1 = U  — surface U coordinate in [0, 1].
            Channel 2 = V  — surface V coordinate in [0, 1].

        Note: Channel 0 is intentionally kept as a raw integer index (cast to
        float32) so that call_iuv() can apply one-hot encoding without
        precision loss from normalisation.
        """
        from densepose.vis.extractor import DensePoseResultExtractor
        extractor = DensePoseResultExtractor()
        instances = outputs

        H, W, _ = entry["image"].shape
        iuv = np.zeros((H, W, 3), dtype=np.float32)

        if not instances.has("pred_densepose"):
            return iuv

        results = extractor(instances)
        if results is None or len(results) == 0:
            return iuv

        data, boxes = results[0]
        # data[0] is a DensePoseChartResult with .labels, .uv
        dp_result = data[0]
        box = boxes[0]
        x1, y1, x2, y2 = [int(v) for v in box.cpu().numpy()]
        bh, bw = y2 - y1, x2 - x1

        # labels: (H_box, W_box) — body part index 0..24 (uint8 on GPU)
        labels = dp_result.labels.cpu().numpy().astype(np.float32)  # (H_box, W_box)
        # uv: (2, H_box, W_box) float32 in [0, 1]
        uv = dp_result.uv.cpu().numpy()  # (2, H_box, W_box)

        # Resize box outputs to (bh, bw) if needed
        if labels.shape != (bh, bw):
            labels = cv2.resize(labels, (bw, bh), interpolation=cv2.INTER_NEAREST)
            u_ch = cv2.resize(uv[0], (bw, bh), interpolation=cv2.INTER_LINEAR)
            v_ch = cv2.resize(uv[1], (bw, bh), interpolation=cv2.INTER_LINEAR)
        else:
            u_ch = uv[0]
            v_ch = uv[1]

        # Clip box to image bounds
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, W), min(y2, H)
        ox1, oy1 = x1c - x1, y1c - y1
        ox2, oy2 = ox1 + (x2c - x1c), oy1 + (y2c - y1c)

        # Store raw integer index in channel 0 (NOT normalised — one-hot is
        # applied later in call_iuv() to avoid float precision issues)
        iuv[y1c:y2c, x1c:x2c, 0] = labels[oy1:oy2, ox1:ox2]   # raw index 0..24
        iuv[y1c:y2c, x1c:x2c, 1] = u_ch[oy1:oy2, ox1:ox2]
        iuv[y1c:y2c, x1c:x2c, 2] = v_ch[oy1:oy2, ox1:ox2]

        return iuv

    def __call__(self, image_or_path, resize=512) -> Image.Image:
        """
        :param image_or_path: Path of the input image.
        :param resize: Resize the input image if its max size is larger than this value.
        :return: Dense pose image.
        """
        # random tmp path with timestamp
        tmp_path = f"./densepose_/tmp/"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        image_path = os.path.join(tmp_path, f"{int(time.time())}-{self.device}-{randint(0, 100000)}.png")
        if isinstance(image_or_path, str):
            assert image_or_path.split(".")[-1] in ["jpg", "png"], "Only support jpg and png images."
            shutil.copy(image_or_path, image_path)
        elif isinstance(image_or_path, Image.Image):
            image_or_path.save(image_path)
        else:
            shutil.rmtree(tmp_path)
            raise TypeError("image_path must be str or PIL.Image.Image")

        output_path = image_path.replace(".png", "_dense.png").replace(".jpg", "_dense.png")
        w, h = Image.open(image_path).size

        file_list = self._get_input_file_list(image_path)
        assert len(file_list), "No input images found!"
        context = self.create_context(self.cfg, output_path)
        for file_name in file_list:
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            # resize
            if (_ := max(img.shape)) > resize:
                scale = resize / _
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

            with torch.no_grad():
                outputs = self.predictor(img)["instances"]
                try:
                    self.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
                except Exception as e:
                    null_gray = Image.new('L', (1, 1))
                    null_gray.save(output_path)

        dense_gray = Image.open(output_path).convert("L")
        dense_gray = dense_gray.resize((w, h), Image.NEAREST)
        # remove image_path and output_path
        os.remove(image_path)
        os.remove(output_path)


        return dense_gray

    def call_iuv(self, image_or_path, resize: int = 512) -> np.ndarray:
        """
        Run DensePose and return a one-hot-encoded IUV tensor as float32.

        The I channel (body-part index 0..24) is expanded into 24 one-hot
        channels using torch.nn.functional.one_hot, then concatenated with
        the U and V channels.

        Args:
            image_or_path: PIL.Image.Image or file path string.
            resize (int): Resize longest side to this before inference.

        Returns:
            np.ndarray shape (H, W, 26) float32.
            Channels  0..23 = one-hot encoding of body-part index (24 classes).
            Channel  24     = U surface coordinate in [0, 1].
            Channel  25     = V surface coordinate in [0, 1].
            Background pixels = 0 in all channels.

        Channel count: 24 (one-hot I) + 1 (U) + 1 (V) = 26.
        """
        tmp_dir = "./densepose_/tmp/"
        os.makedirs(tmp_dir, exist_ok=True)

        image_path = os.path.join(tmp_dir, f"{int(time.time())}-{self.device}-{randint(0, 100000)}.png")
        if isinstance(image_or_path, str):
            assert image_or_path.split(".")[-1].lower() in ("jpg", "jpeg", "png")
            shutil.copy(image_or_path, image_path)
        elif isinstance(image_or_path, Image.Image):
            image_or_path.save(image_path)
        else:
            raise TypeError("image_or_path must be str or PIL.Image.Image")

        orig_w, orig_h = Image.open(image_path).size

        img = read_image(image_path, format="BGR")
        if (_ := max(img.shape)) > resize:
            scale = resize / _
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        with torch.no_grad():
            outputs = self.predictor(img)["instances"]

        try:
            # raw_iuv: (H_inf, W_inf, 3) — channel 0 is raw integer index 0..24
            raw_iuv = self.execute_on_outputs_iuv({"image": img}, outputs)
        except Exception:
            raw_iuv = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)

        # ── Resize to original image dimensions ─────────────────────────
        inf_h, inf_w = raw_iuv.shape[:2]
        if (inf_h, inf_w) != (orig_h, orig_w):
            # I channel: nearest-neighbour to preserve integer indices
            i_resized = cv2.resize(
                raw_iuv[:, :, 0], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            # U, V channels: bilinear
            u_resized = cv2.resize(
                raw_iuv[:, :, 1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
            v_resized = cv2.resize(
                raw_iuv[:, :, 2], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            i_resized = raw_iuv[:, :, 0]
            u_resized = raw_iuv[:, :, 1]
            v_resized = raw_iuv[:, :, 2]

        # ── One-hot encode I channel (24 classes, indices 0..23) ─────────
        # DensePose labels: 0 = background, 1..24 = body parts.
        # We clamp to [0, 23] so index 24 maps to class 23 (rare edge case).
        i_int = torch.from_numpy(i_resized).long().clamp(0, 23)  # (H, W)

        # torch.nn.functional.one_hot: (H, W) → (H, W, 24), no loops
        i_onehot = torch.nn.functional.one_hot(i_int, num_classes=24)  # (H, W, 24)
        i_onehot = i_onehot.float()                                      # float32

        # ── Concatenate: [one-hot I (24), U (1), V (1)] → (H, W, 26) ────
        u_tensor = torch.from_numpy(u_resized).unsqueeze(-1)  # (H, W, 1)
        v_tensor = torch.from_numpy(v_resized).unsqueeze(-1)  # (H, W, 1)

        iuv26 = torch.cat([i_onehot, u_tensor, v_tensor], dim=-1)  # (H, W, 26)

        os.remove(image_path)
        return iuv26.numpy()  # (H, W, 26) float32


if __name__ == '__main__':
    pass
