"""
Comprehensive test suite for GIS interpretation layer (Step 5).
Verifies GeoTIFF parsing, channel selection, alignment check, polygonization,
geodesic area calculation, GeoJSON assembly, and end-to-end LangGraph integration.
"""
from __future__ import annotations
import os
import io
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds, from_origin
from gis.raster import GeoTIFFReader, GeoTIFFMetadata, AlignmentChecker
from gis.processor import GISProcessor
from tools.change_detection import ChangeDetectionTool
from ai.graph.builder import build_graph


@pytest.fixture
def sample_geotiff_bytes():
    """Generates a small in-memory 3-band GeoTIFF in EPSG:32643 (UTM 43N meters)."""
    width, height = 64, 64
    transform = from_origin(775000, 1435000, 10.0, 10.0)  # 10m resolution
    crs = "EPSG:32643"

    data = np.random.randint(50, 200, size=(3, height, width), dtype=np.uint8)

    buf = io.BytesIO()
    with rasterio.open(
        buf, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype='uint8',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)

    return buf.getvalue()


# 1. GeoTIFF metadata extraction
def test_1_geotiff_metadata_extraction(sample_geotiff_bytes):
    meta = GeoTIFFReader.read_metadata(sample_geotiff_bytes)
    assert meta.width == 64
    assert meta.height == 64
    assert meta.band_count == 3
    assert "32643" in meta.crs
    assert meta.resolution == (10.0, 10.0)
    assert len(meta.bounds) == 4
    assert len(meta.transform) == 6


# 2. CRS, Transform, Bounds extraction
def test_2_crs_transform_bounds_integrity(sample_geotiff_bytes):
    meta = GeoTIFFReader.read_metadata(sample_geotiff_bytes)
    assert meta.bounds[0] < meta.bounds[2]  # minx < maxx
    assert meta.bounds[1] < meta.bounds[3]  # miny < maxy
    assert meta.transform[0] == 10.0       # pixel width


# 3. RGB Band Selection Policy
def test_3_channel_policy_4_band_and_s2():
    # 4-band test
    buf4 = io.BytesIO()
    with rasterio.open(
        buf4, 'w', driver='GTiff', height=10, width=10, count=4, dtype='uint8', crs="EPSG:4326"
    ) as dst:
        dst.write(np.ones((4, 10, 10), dtype=np.uint8) * 100)
    rgb4, meta4 = GeoTIFFReader.read_rgb(buf4.getvalue())
    assert rgb4.shape == (10, 10, 3)

    # 1-band test
    buf1 = io.BytesIO()
    with rasterio.open(
        buf1, 'w', driver='GTiff', height=10, width=10, count=1, dtype='uint8', crs="EPSG:4326"
    ) as dst:
        dst.write(np.ones((1, 10, 10), dtype=np.uint8) * 128)
    rgb1, meta1 = GeoTIFFReader.read_rgb(buf1.getvalue())
    assert rgb1.shape == (10, 10, 3)


# 4. Incompatible Raster Grids
def test_4_incompatible_raster_grids():
    meta1 = GeoTIFFMetadata(64, 64, "EPSG:4326", [0, 1, 0, 0, 0, -1], [77, 12, 78, 13], (1, 1), 3)
    meta2 = GeoTIFFMetadata(128, 128, "EPSG:32643", [0, 10, 0, 0, 0, -10], [100, 200, 300, 400], (10, 10), 3)

    ok, errors = AlignmentChecker.verify_alignment(meta1, meta2)
    assert ok is False
    assert len(errors) >= 2


# 5. Mask-to-Polygon Conversion & Min Pixel Threshold
def test_5_polygonization_and_filtering():
    processor = GISProcessor(min_polygon_pixels=5)
    mask = np.zeros((50, 50), dtype=np.uint8)
    # Add a small 2-pixel noise cluster (should be filtered out)
    mask[2:4, 2] = 1
    # Add a 16-pixel change cluster (should be preserved)
    mask[10:14, 10:14] = 1

    transform = [77.0, 0.001, 0.0, 13.0, 0.0, -0.001]
    fc, summary = processor.polygonize_change_mask(mask, transform, src_crs="EPSG:4326")

    assert summary["polygon_count"] == 1
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["properties"]["change_type"] == "change_detected"
    assert feat["properties"]["pixel_count"] >= 5


# 6. Geodesic Metric Area Calculation
def test_6_area_calculation_accuracy():
    processor = GISProcessor(min_polygon_pixels=1)
    mask = np.ones((10, 10), dtype=np.uint8)
    # 10x10 pixels at 10m resolution in UTM = 100m x 100m = 10,000 m2 = 1.0 ha
    transform = from_origin(500000, 1000000, 10.0, 10.0)
    fc, summary = processor.polygonize_change_mask(mask, transform, src_crs="EPSG:32643")

    assert summary["total_changed_area_m2"] == 10000.0
    assert summary["total_changed_area_ha"] == 1.0
    assert fc["features"][0]["properties"]["area_ha"] == 1.0


# 7. GeoJSON Validity
def test_7_geojson_structure():
    processor = GISProcessor(min_polygon_pixels=1)
    mask = np.ones((5, 5), dtype=np.uint8)
    transform = [77.5, 0.0001, 0.0, 12.9, 0.0, -0.0001]
    fc, _ = processor.polygonize_change_mask(mask, transform, src_crs="EPSG:4326")

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    props = fc["features"][0]["properties"]
    assert "area_m2" in props
    assert "area_ha" in props
    assert props["change_type"] == "change_detected"
    assert props["confidence_status"] == "uncalibrated"


# 8. Null/Uncalibrated Confidence Handling in T4_Change
def test_8_null_uncalibrated_confidence():
    tool = ChangeDetectionTool(mode="real")
    with open("backend/real_data/opt_0611.png", "rb") as f0, open("backend/real_data/opt_0810.png", "rb") as f1:
        b0, b1 = f0.read(), f1.read()

    res = tool.run(image_bytes_t0=b0, image_bytes_t1=b1, mode="real")
    assert res["confidence"] is None
    assert res["confidence_status"] == "uncalibrated"
    assert res["evidence"][0]["label"] == "change_detected"


# 9. End-to-End LangGraph with Real GeoTIFF -> GIS -> GeoJSON
def test_9_end_to_end_geotiff_langgraph():
    # Build 2 real synthetic GeoTIFFs
    width, height = 128, 128
    transform = from_origin(77.5, 13.0, 0.0001, 0.0001)

    t0_data = np.random.randint(40, 180, size=(3, height, width), dtype=np.uint8)
    t1_data = t0_data.copy()
    t1_data[:, 20:50, 20:50] = 255  # Distinct block of change

    b0_buf = io.BytesIO()
    with rasterio.open(b0_buf, 'w', driver='GTiff', height=height, width=width, count=3, dtype='uint8', crs="EPSG:4326", transform=transform) as d0:
        d0.write(t0_data)

    b1_buf = io.BytesIO()
    with rasterio.open(b1_buf, 'w', driver='GTiff', height=height, width=width, count=3, dtype='uint8', crs="EPSG:4326", transform=transform) as d1:
        d1.write(t1_data)

    graph = build_graph()
    state = {
        "session_id": "geotiff-e2e",
        "query": "Identify land cover changes between 2020 and 2024.",
        "image_bytes": [b0_buf.getvalue(), b1_buf.getvalue()],
        "image_modalities": ["optical", "optical"],
        "image_filenames": ["t0.tif", "t1.tif"],
        "metadata": {"mode": "real"}
    }

    out = graph.invoke(state)
    assert out.get("intent") == "change"
    assert out.get("selected_tool") == "T4_Change"
    assert out.get("geojson") is not None
    assert out["geojson"]["type"] == "FeatureCollection"
    assert len(out["geojson"]["features"]) > 0

    # Verify semantic label is change_detected, not new_construction
    for feat in out["geojson"]["features"]:
        assert feat["properties"]["change_type"] == "change_detected"
