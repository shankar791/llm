"""
Coordinate reference system utilities for SatQuery AI.

Handles reprojection between common satellite imagery CRS and WGS-84 (EPSG:4326),
which is what Leaflet expects.
"""
from __future__ import annotations


def reproject_bbox(
    bbox: list[float],
    src_crs: str,
    dst_crs: str = "EPSG:4326",
) -> list[float]:
    """
    Reproject a bounding box from src_crs to dst_crs.

    Args:
        bbox: [minx, miny, maxx, maxy] in src_crs units.
        src_crs: Source CRS as EPSG string, e.g. 'EPSG:32633'.
        dst_crs: Target CRS. Defaults to WGS-84.

    Returns:
        [minx, miny, maxx, maxy] in dst_crs units.

    Phase 1 implementation: use pyproj.Transformer.from_crs().
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        minx, miny, maxx, maxy = bbox
        p1_x, p1_y = transformer.transform(minx, miny)
        p2_x, p2_y = transformer.transform(maxx, maxy)
        return [min(p1_x, p2_x), min(p1_y, p2_y), max(p1_x, p2_x), max(p1_y, p2_y)]
    except ImportError:
        raise NotImplementedError(
            "reproject_bbox() requires pyproj. Install with: pip install pyproj"
        )


def gdal_geotransform_to_bbox(geotransform: list[float], width: int, height: int) -> list[float]:
    """
    Convert a GDAL GeoTransform to a geographic bounding box.

    Args:
        geotransform: 6-element list [originX, pixelW, 0, originY, 0, pixelH].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        [west, south, east, north] in the image's native CRS.
    """
    origin_x, pixel_w, _, origin_y, _, pixel_h = geotransform
    west = origin_x
    east = origin_x + width * pixel_w
    north = origin_y
    south = origin_y + height * pixel_h  # pixel_h is negative for north-up images
    return [west, min(south, north), east, max(south, north)]
