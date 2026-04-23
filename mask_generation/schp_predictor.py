"""
SCHP (Self-Correction for Human Parsing) predictor — standalone copy.

Given a person image (PIL.Image or file path), runs the SCHP ResNet-101
model and returns a palette-indexed PIL.Image with clothing/body-part labels.

Supports two dataset variants detected automatically from the checkpoint name:
  - 'lip'  : 20 classes  (exp-schp-201908261155-lip.pth)
  - 'atr'  : 18 classes  (exp-schp-201908301523-atr.pth)
  - 'pascal': 7 classes

Checkpoint download:
  LIP : https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH
  ATR : https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0y-OIXHD9usnXm9
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------

DATASET_SETTINGS = {
    "lip": {
        "input_size": [473, 473],
        "num_classes": 20,
        "label": [
            "Background", "Hat", "Hair", "Glove", "Sunglasses",
            "Upper-clothes", "Dress", "Coat", "Socks", "Pants",
            "Jumpsuits", "Scarf", "Skirt", "Face", "Left-arm",
            "Right-arm", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe",
        ],
    },
    "atr": {
        "input_size": [512, 512],
        "num_classes": 18,
        "label": [
            "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes",
            "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe",
            "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
            "Bag", "Scarf",
        ],
    },
    "pascal": {
        "input_size": [512, 512],
        "num_classes": 7,
        "label": [
            "Background", "Head", "Torso", "Upper Arms",
            "Lower Arms", "Upper Legs", "Lower Legs",
        ],
    },
}


# ---------------------------------------------------------------------------
# Palette helper
# ---------------------------------------------------------------------------

def _get_palette(num_cls: int) -> list:
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(n):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            lab >>= 3
    return palette


# ---------------------------------------------------------------------------
# Affine transform helpers (copied from SCHP/utils/transforms.py)
# ---------------------------------------------------------------------------

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad: float):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    ]


def _get_affine_transform(
    center, scale, rot, output_size,
    shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, (np.ndarray, list)):
        scale = np.array([scale, scale])
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size[1], output_size[0]
    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _transform_logits(logits, center, scale, width, height, input_size):
    trans = _get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        target_logits.append(target_logit)
    return np.stack(target_logits, axis=2)


# ---------------------------------------------------------------------------
# SCHP network (AugmentCE2P ResNet-101) — inline copy
# ---------------------------------------------------------------------------

import torch.nn as nn
from torch.nn import functional as F
from torch.nn import BatchNorm2d, LeakyReLU

affine_par = True


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False,
        )
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu_inplace(out + residual)


class _PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, s) for s in sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_features),
            LeakyReLU(),
        )

    def _make_stage(self, features, out_features, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            BatchNorm2d(out_features),
            LeakyReLU(),
        )

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(s(feats), size=(h, w), mode="bilinear", align_corners=True) for s in self.stages]
        return self.bottleneck(torch.cat(priors + [feats], 1))


class _EdgeModule(nn.Module):
    def __init__(self, in_fea=(256, 512, 1024), mid_fea=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_fea[0], mid_fea, 1, bias=False), BatchNorm2d(mid_fea), LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_fea[1], mid_fea, 1, bias=False), BatchNorm2d(mid_fea), LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_fea[2], mid_fea, 1, bias=False), BatchNorm2d(mid_fea), LeakyReLU())
        self.conv4 = nn.Conv2d(mid_fea, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        e1 = self.conv1(x1)
        e2 = F.interpolate(self.conv2(x2), size=(h, w), mode="bilinear", align_corners=True)
        e3 = F.interpolate(self.conv3(x3), size=(h, w), mode="bilinear", align_corners=True)
        return torch.cat([e1, e2, e3], dim=1)


class _DecoderModule(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), BatchNorm2d(256), LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), BatchNorm2d(48), LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, 1, bias=False), BatchNorm2d(256), LeakyReLU(),
            nn.Conv2d(256, 256, 1, bias=False), BatchNorm2d(256), LeakyReLU(),
        )

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode="bilinear", align_corners=True)
        xl = self.conv2(xl)
        return self.conv3(torch.cat([xt, xl], dim=1))


class _SCHPResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super().__init__()
        self.conv1 = _conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = _conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = _conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = _PSPModule(2048, 512)
        self.edge = _EdgeModule()
        self.decoder = _DecoderModule(num_classes)
        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False), BatchNorm2d(256), LeakyReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1, bias=True),
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        layers = []
        gen_mg = lambda idx, grids: grids[idx % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=gen_mg(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=gen_mg(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.context_encoding(x5)
        parsing_fea = self.decoder(x, x2)
        edge_fea = self.edge(x2, x3, x4)
        fusion_result = self.fushion(torch.cat([parsing_fea, edge_fea], dim=1))
        return fusion_result


def _build_schp_model(num_classes: int) -> _SCHPResNet:
    return _SCHPResNet(_Bottleneck, [3, 4, 23, 3], num_classes)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SCHPPredictor:
    """
    Runs SCHP human parsing on a single image.

    Args:
        ckpt_path (str): Path to the .pth checkpoint file.
            The dataset type (lip / atr / pascal) is inferred from the filename.
        device (str): 'cuda' or 'cpu'
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        dataset_type = None
        for key in ("lip", "atr", "pascal"):
            if key in ckpt_path.lower():
                dataset_type = key
                break
        assert dataset_type is not None, (
            f"Cannot infer dataset type from checkpoint path: {ckpt_path!r}. "
            "Filename must contain 'lip', 'atr', or 'pascal'."
        )

        self.device = device
        settings = DATASET_SETTINGS[dataset_type]
        self.num_classes = settings["num_classes"]
        self.input_size = settings["input_size"]
        self.aspect_ratio = self.input_size[1] / self.input_size[0]
        self.palette = _get_palette(self.num_classes)
        self.label = settings["label"]

        self.model = _build_schp_model(self.num_classes).to(device)
        self._load_ckpt(ckpt_path)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
        ])
        self.upsample = torch.nn.Upsample(size=self.input_size, mode="bilinear", align_corners=True)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_ckpt(self, ckpt_path: str) -> None:
        rename_map = {
            "decoder.conv3.2.weight": "decoder.conv3.3.weight",
            "decoder.conv3.3.weight": "decoder.conv3.4.weight",
            "decoder.conv3.3.bias":   "decoder.conv3.4.bias",
            "decoder.conv3.3.running_mean": "decoder.conv3.4.running_mean",
            "decoder.conv3.3.running_var":  "decoder.conv3.4.running_var",
            "fushion.3.weight": "fushion.4.weight",
            "fushion.3.bias":   "fushion.4.bias",
        }
        raw = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        # strip 'module.' prefix added by DataParallel
        stripped = OrderedDict((k[7:], v) for k, v in raw.items())
        renamed = OrderedDict(
            (rename_map.get(k, k), v) for k, v in stripped.items()
        )
        self.model.load_state_dict(renamed, strict=False)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        return center, np.array([w, h], dtype=np.float32)

    def _preprocess(self, image: Union[str, Image.Image]):
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = np.array(image)

        h, w, _ = img.shape
        center, scale = self._box2cs([0, 0, w - 1, h - 1])
        trans = _get_affine_transform(center, scale, 0, self.input_size)
        warped = cv2.warpAffine(
            img, trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        tensor = self.transform(warped).to(self.device).unsqueeze(0)
        meta = {"center": center, "height": h, "width": w, "scale": scale, "rotation": 0}
        return tensor, meta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image_or_path: Union[str, Image.Image, list]) -> Union[Image.Image, list]:
        """
        Run SCHP parsing on one image or a list of images.

        Args:
            image_or_path: A file path, PIL.Image, or list of either.

        Returns:
            A palette-indexed PIL.Image (or list of them for batch input).
            Each pixel value is a class index; use result.getpalette() for colours.
        """
        if isinstance(image_or_path, list):
            tensors, metas = [], []
            for img in image_or_path:
                t, m = self._preprocess(img)
                tensors.append(t)
                metas.append(m)
            batch = torch.cat(tensors, dim=0)
        else:
            batch, meta = self._preprocess(image_or_path)
            metas = [meta]

        with torch.no_grad():
            output = self.model(batch)

        upsampled = self.upsample(output).permute(0, 2, 3, 1)  # BCHW → BHWC

        results = []
        for upsample_output, meta in zip(upsampled, metas):
            logits = _transform_logits(
                upsample_output.cpu().numpy(),
                meta["center"], meta["scale"],
                meta["width"], meta["height"],
                input_size=self.input_size,
            )
            parsing = np.argmax(logits, axis=2).astype(np.uint8)
            out_img = Image.fromarray(parsing)
            out_img.putpalette(self.palette)
            results.append(out_img)

        return results[0] if len(results) == 1 else results
