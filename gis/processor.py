"""
GISProcessor — Deterministic geospatial vectorization, polygonization, area calculation,
and GeoJSON FeatureCollection generation for SatQuery AI.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import pyproj


class GISProcessor:
    """
    Deterministic geospatial engine converting binary raster masks into
    calibrated GeoJSON polygons and calculating precise metric areas.
    """

    MIN_POLYGON_PIXELS_DEFAULT = 5

    def __init__(self, min_polygon_pixels: int = MIN_POLYGON_PIXELS_DEFAULT):
        self.min_polygon_pixels = min_polygon_pixels
        self._geod = pyproj.Geod(ellps="WGS84")

    def polygonize_change_mask(
        self,
        change_mask: np.ndarray,
        transform: Union[List[float], Affine],
        src_crs: str = "EPSG:4326",
        min_polygon_pixels: Optional[int] = None,
        properties_template: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert a binary 2D change mask into a GeoJSON FeatureCollection.

        Args:
            change_mask: 2D numpy array (H, W) uint8 where 1 indicates change
            transform: Affine transform [a, b, c, d, e, f], GDAL geotransform, or Affine object
            src_crs: Source Coordinate Reference System
            min_polygon_pixels: Minimum pixel count threshold to filter out noise
            properties_template: Contextual properties (dates, model name, etc.)

        Returns:
            Tuple of (geojson_feature_collection, summary_metrics)
        """
        if change_mask.ndim != 2:
            raise ValueError(f"change_mask must be 2D, got shape {change_mask.shape}")

        min_px = min_polygon_pixels if min_polygon_pixels is not None else self.min_polygon_pixels

        # Ensure Affine object
        if isinstance(transform, Affine):
            aff = transform
        elif isinstance(transform, (list, tuple)):
            if len(transform) != 6:
                raise ValueError(f"Transform must have 6 elements, got {len(transform)}")
            # Distinguish GDAL [c, a, b, f, d, e] vs Affine [a, b, c, d, e, f]
            if abs(transform[0]) > abs(transform[1]) and abs(transform[1]) > 0:
                aff = Affine.from_gdal(*transform)
            else:
                aff = Affine(*transform)
        else:
            raise TypeError(f"Unsupported transform type: {type(transform)}")

        # Ensure uint8 binary mask
        mask_uint8 = (change_mask > 0).astype(np.uint8)
        total_pixels = int(mask_uint8.size)
        changed_pixels = int(np.count_nonzero(mask_uint8))
        change_fraction = round(float(changed_pixels / total_pixels), 4) if total_pixels > 0 else 0.0

        # Extract shapes from mask where value == 1
        shapes_gen = rasterio.features.shapes(mask_uint8, mask=(mask_uint8 == 1), transform=aff)

        # Coordinate transformer to WGS-84 (EPSG:4326) if needed
        is_geographic = "4326" in src_crs or "WGS" in src_crs.upper()
        if not is_geographic:
            transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        else:
            transformer = None

        pixel_area_geo = abs(aff.a * aff.e) if (aff.a * aff.e) != 0 else 1.0

        features = []
        total_changed_area_m2 = 0.0
        feature_idx = 1

        template = properties_template or {}
        before_date = template.get("before_date") or template.get("acquisition_dates", {}).get("t0")
        after_date = template.get("after_date") or template.get("acquisition_dates", {}).get("t1")
        source_model = template.get("model", "ChangeFormer")
        confidence_status = template.get("confidence_status", "uncalibrated")

        for geom_dict, val in shapes_gen:
            if int(val) != 1:
                continue

            geom = shape(geom_dict)
            if geom.is_empty or not geom.is_valid:
                geom = geom.buffer(0)
                if geom.is_empty:
                    continue

            # Compute pixel count accurately from geom.area / pixel_area_geo
            px_count = max(1, int(round(abs(geom.area) / pixel_area_geo)))

            # Filter noise below minimum threshold
            if px_count < min_px:
                continue

            # Calculate accurate metric area in m2
            if is_geographic:
                try:
                    area_m2_val, _ = self._geod.geometry_area_perimeter(geom)
                    area_m2 = abs(area_m2_val) if not math.isnan(area_m2_val) else 0.0
                except Exception:
                    area_m2 = 0.0

                if area_m2 <= 0.0:
                    # Robust geodesic approximation at latitude
                    lat = geom.centroid.y
                    deg_lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
                    deg_lon_m = 111412.84 * math.cos(math.radians(lat))
                    area_m2 = abs(geom.area) * deg_lat_m * deg_lon_m
            else:
                # Projected CRS in meters
                area_m2 = abs(geom.area)

            area_ha = round(area_m2 / 10_000.0, 4)
            area_m2 = round(area_m2, 2)

            # Transform geometry to EPSG:4326 for standard GeoJSON output
            if transformer is not None:
                from shapely.ops import transform as shp_transform
                geom_4326 = shp_transform(transformer.transform, geom)
            else:
                geom_4326 = geom

            feature = {
                "type": "Feature",
                "id": f"change_{feature_idx:04d}",
                "geometry": mapping(geom_4326),
                "properties": {
                    "feature_id": feature_idx,
                    "area_m2": area_m2,
                    "area_ha": area_ha,
                    "pixel_count": px_count,
                    "source_model": source_model,
                    "before_date": before_date,
                    "after_date": after_date,
                    "change_type": "change_detected",
                    "confidence_status": confidence_status
                }
            }
            features.append(feature)
            total_changed_area_m2 += area_m2
            feature_idx += 1

        feature_collection = {
            "type": "FeatureCollection",
            "features": features
        }

        summary_metrics = {
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "change_fraction": change_fraction,
            "polygon_count": len(features),
            "total_changed_area_m2": round(total_changed_area_m2, 2),
            "total_changed_area_ha": round(total_changed_area_m2 / 10_000.0, 4),
            "min_polygon_pixels_threshold": min_px,
            "crs": src_crs
        }

        return feature_collection, summary_metrics
