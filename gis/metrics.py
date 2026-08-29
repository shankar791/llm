"""
Geospatial metric calculations for SatQuery AI.

All functions operate on either pixel-space or geographic coordinates.
"""
from __future__ import annotations
import math


def area_ha(bbox_geo: list[float]) -> float:
    """
    Compute the approximate area of a geographic bounding box in hectares.

    Uses the equirectangular approximation — accurate within ~1% for
    bounding boxes smaller than ~100 km.

    Args:
        bbox_geo: [west, south, east, north] in decimal degrees.

    Returns:
        Area in hectares.
    """
    west, south, east, north = bbox_geo
    lat_m = math.radians((north + south) / 2)
    dx_m = (east - west) * 111_320 * math.cos(lat_m)
    dy_m = (north - south) * 110_540
    return abs(dx_m * dy_m) / 10_000  # m² → ha


def overlap_iou(box_a: list[int], box_b: list[int]) -> float:
    """
    Intersection-over-Union of two pixel bounding boxes.

    Args:
        box_a: [x0, y0, x1, y1]
        box_b: [x0, y0, x1, y1]

    Returns:
        IoU in [0, 1]. Returns 0.0 if boxes do not overlap.
    """
    xa0, ya0, xa1, ya1 = box_a
    xb0, yb0, xb1, yb1 = box_b
    inter_x = max(0, min(xa1, xb1) - max(xa0, xb0))
    inter_y = max(0, min(ya1, yb1) - max(ya0, yb0))
    inter = inter_x * inter_y
    area_a = (xa1 - xa0) * (ya1 - ya0)
    area_b = (xb1 - xb0) * (yb1 - yb0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def change_fraction_to_area(change_fraction: float, image_extent_ha: float) -> float:
    """
    Convert a fractional change mask to changed area in hectares.

    Args:
        change_fraction: Fraction of pixels detected as changed, in [0, 1].
        image_extent_ha: Total image footprint in hectares.

    Returns:
        Changed area in hectares.
    """
    return change_fraction * image_extent_ha
