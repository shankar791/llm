"""
Standalone Unit & Integration Test Suite for ChangeFormer (Official Architecture).
Tests Step 4B real checkpoint loading, strict key compatibility, and deterministic inference.
"""
from __future__ import annotations
import os
import pytest
import numpy as np
import torch
from PIL import Image
from models.changeformer.adapter import ChangeFormerAdapter
from models.changeformer.network import ChangeFormerModel, ChangeFormer


# 1. Real Checkpoint Loading & Self-Test
def test_1_real_checkpoint_loading_and_zero_key_mismatch():
    """Verify official ChangeFormer pretrained checkpoint loads with 0 missing and 0 unexpected keys."""
    adapter = ChangeFormerAdapter(mode="real")
    load_info = adapter.load()

    assert load_info["mode"] == "real"
    assert load_info["is_mock"] is False
    assert load_info["missing_keys_count"] == 0
    assert load_info["unexpected_keys_count"] == 0
    assert load_info["total_parameters"] == 41_026_674
    assert load_info["loaded_parameters"] == 41_026_674
    assert adapter._model is not None
    assert not adapter._model.training


# 2. Strict Checkpoint Key Compatibility
def test_2_strict_state_dict_key_compatibility():
    """Verify all 373 checkpoint keys map directly to ChangeFormerModel with strict=True."""
    import safetensors.torch
    ckpt_path = ChangeFormerAdapter.DEFAULT_CHECKPOINT
    assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"

    state_dict = safetensors.torch.load_file(ckpt_path)
    model = ChangeFormerModel()
    load_res = model.load_state_dict(state_dict, strict=True)
    assert len(load_res.missing_keys) == 0
    assert len(load_res.unexpected_keys) == 0


# 3. Real Pretrained Inference on Real Data
def test_3_real_pretrained_inference_execution():
    """Verify real inference on real bi-temporal optical images generates valid 2D change mask."""
    adapter = ChangeFormerAdapter(mode="real")
    adapter.load()

    opt1_path = "backend/real_data/opt_0611.png"
    opt2_path = "backend/real_data/opt_0810.png"
    assert os.path.exists(opt1_path) and os.path.exists(opt2_path)

    img_t0 = Image.open(opt1_path)
    img_t1 = Image.open(opt2_path)

    res = adapter.detect(img_t0, img_t1)
    assert res["status"] == "success"
    assert "change_mask" in res
    assert isinstance(res["change_mask"], np.ndarray)
    assert res["change_mask"].shape == (651, 760)
    assert res["change_mask"].dtype == np.uint8

    meta = res["metadata"]
    assert meta["model"] == "ChangeFormer"
    assert meta["mode"] == "real"
    assert meta["is_mock"] is False
    assert meta["inference_time_ms"] > 0
    assert 0.0 < meta["change_fraction"] < 1.0


# 4. Explicit Failure on Missing Real Checkpoint
def test_4_explicit_failure_when_real_checkpoint_missing():
    """Verify requesting real mode with nonexistent checkpoint raises FileNotFoundError."""
    adapter = ChangeFormerAdapter(checkpoint_path="/nonexistent/path/weights.safetensors", mode="real")
    with pytest.raises(FileNotFoundError, match="Real ChangeFormer inference requires a valid checkpoint path"):
        adapter.load()


# 5. Mock Mode Remains Isolated and Explicitly Tagged
def test_5_mock_mode_isolated_and_tagged():
    """Verify mock mode initializes cleanly without checkpoint and is marked is_mock=True."""
    adapter = ChangeFormerAdapter(mode="mock")
    load_info = adapter.load()

    assert load_info["mode"] == "mock"
    assert load_info["is_mock"] is True
    assert load_info["checkpoint"] == "mock_initialized"

    t0 = np.zeros((256, 256, 3), dtype=np.uint8)
    t1 = np.ones((256, 256, 3), dtype=np.uint8) * 255
    res = adapter.detect(t0, t1)
    assert res["metadata"]["is_mock"] is True
    assert res["metadata"]["mode"] == "mock"


# 6. Strict Determinism
def test_6_inference_is_strictly_deterministic():
    """Verify repeated inference with identical inputs returns identical change masks."""
    adapter = ChangeFormerAdapter(mode="real")
    adapter.load()

    t0 = np.zeros((100, 100, 3), dtype=np.uint8)
    t1 = np.ones((100, 100, 3), dtype=np.uint8) * 200

    res1 = adapter.detect(t0, t1)
    res2 = adapter.detect(t0, t1)

    assert np.array_equal(res1["change_mask"], res2["change_mask"])


# 7. Input Validation & Missing Images
def test_7_missing_images_and_invalid_shapes():
    """Verify missing inputs or unsupported tensor ranks raise ValueError."""
    adapter = ChangeFormerAdapter(mode="mock")
    adapter.load()

    valid_img = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Both image_t0 and image_t1 are strictly required"):
        adapter.detect(valid_img, None)

    invalid_shape = np.zeros((256,), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported numpy array dimensions"):
        adapter.detect(invalid_shape, valid_img)
