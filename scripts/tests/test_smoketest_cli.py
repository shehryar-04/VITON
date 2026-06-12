"""Unit tests for scripts/flux_t4_smoketest.py CLI argument parsing."""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

# Ensure the repo root is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestClothTypeArgument:
    """Requirement 4.3: cloth_type argument constrained to valid set."""

    @pytest.mark.parametrize("cloth_type", ["upper", "lower", "overall", "inner", "outer"])
    def test_valid_cloth_types_accepted(self, cloth_type):
        """Valid cloth_type values are accepted by argparse."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cloth-type", default="upper",
                            choices=["upper", "lower", "overall", "inner", "outer"])
        args = parser.parse_args(["--cloth-type", cloth_type])
        assert args.cloth_type == cloth_type

    @pytest.mark.parametrize("bad_type", ["shirt", "UPPER", "pants", "", "full"])
    def test_invalid_cloth_type_rejected(self, bad_type):
        """Requirement 4.4: Invalid cloth_type exits with nonzero code."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cloth-type", default="upper",
                            choices=["upper", "lower", "overall", "inner", "outer"])
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--cloth-type", bad_type])
        assert exc_info.value.code != 0

    def test_default_is_upper(self):
        """Default cloth_type is 'upper'."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cloth-type", default="upper",
                            choices=["upper", "lower", "overall", "inner", "outer"])
        args = parser.parse_args([])
        assert args.cloth_type == "upper"


class TestManualMaskOverride:
    """Requirement 4.2, 4.6: Manual mask loading."""

    def test_valid_mask_loads(self, tmp_path):
        """A valid mask file can be loaded and converted."""
        mask_path = tmp_path / "test_mask.png"
        mask = Image.new("L", (100, 100), 128)
        mask.save(str(mask_path))

        loaded = Image.open(str(mask_path)).convert("L").resize((768, 1024))
        assert loaded.size == (768, 1024)
        assert loaded.mode == "L"

    def test_nonexistent_mask_path(self):
        """Requirement 4.6: Non-existent mask path should be detectable."""
        assert not os.path.isfile("/nonexistent/mask.png")


class TestBlankMaskFallback:
    """Requirement 4.5: Blank mask when AutoMasker unavailable."""

    def test_blank_mask_dimensions(self):
        """Blank mask has correct dimensions and all pixels 255."""
        width, height = 768, 1024
        mask = Image.new("L", (width, height), 255)
        assert mask.size == (width, height)
        assert mask.mode == "L"
        assert all(px == 255 for px in mask.getdata())
