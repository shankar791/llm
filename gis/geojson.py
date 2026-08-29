"""
GeoJSON construction utilities for SatQuery AI.

Converts pixel-space bounding boxes from tool outputs to GeoJSON Features
that Leaflet can render as overlays on the satellite basemap.
"""
from __future__ import annotations
from typing import Optional


class GeoJSONBuilder:
    """
    Convert pixel-space bounding boxes to GeoJSON for Leaflet rendering.

    Pixel coordinates are transformed to geographic coordinates using the
    image's affine transform (GDAL GeoTransform) when available, or via
    a linear interpolation between provided corner coordinates.
    """

    def bbox_to_feature(
        self,
        bbox_pixels: list[int],
        transform: Optional[list] = None,
        image_shape: Optional[tuple[int, int]] = None,
        corner_coords: Optional[tuple[float, float, float, float]] = None,
        label: str = "",
        properties: Optional[dict] = None,
    ) -> dict:
        """
        Convert a pixel bounding box to a GeoJSON Feature (Polygon).

        Args:
            bbox_pixels: [x0, y0, x1, y1] in pixel coordinates.
            transform: GDAL GeoTransform [originX, pixelW, 0, originY, 0, pixelH].
            image_shape: (height, width) in pixels — required if transform is None.
            corner_coords: (west, south, east, north) in decimal degrees —
                           used for linear interpolation when no transform is given.
            label: Human-readable label for the feature.
            properties: Additional GeoJSON properties to include.

        Returns:
            GeoJSON Feature dict with geometry type 'Polygon'.

        Raises:
            ValueError: if insufficient coordinate info is provided.
        """
        if transform is None and corner_coords is None:
            raise ValueError(
                "Provide either 'transform' (GDAL GeoTransform) or "
                "'corner_coords' (west, south, east, north)."
            )

        if transform is not None:
            # GDAL affine transform: pixel (col, row) → (lon, lat)
            x0, y0, x1, y1 = bbox_pixels

            def px_to_geo(col: int, row: int) -> list[float]:
                lon = transform[0] + col * transform[1] + row * transform[2]
                lat = transform[3] + col * transform[4] + row * transform[5]
                return [lon, lat]

            coords = [
                px_to_geo(x0, y0),
                px_to_geo(x1, y0),
                px_to_geo(x1, y1),
                px_to_geo(x0, y1),
                px_to_geo(x0, y0),  # close ring
            ]
        else:
            # Linear interpolation between image corners
            if image_shape is None:
                raise ValueError("image_shape (height, width) is required when using corner_coords")
            west, south, east, north = corner_coords
            H, W = image_shape
            x0, y0, x1, y1 = bbox_pixels
            lon0 = west + (x0 / W) * (east - west)
            lon1 = west + (x1 / W) * (east - west)
            lat0 = north - (y0 / H) * (north - south)
            lat1 = north - (y1 / H) * (north - south)
            coords = [
                [lon0, lat0],
                [lon1, lat0],
                [lon1, lat1],
                [lon0, lat1],
                [lon0, lat0],
            ]

        props = {"label": label}
        if properties:
            props.update(properties)

        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": props,
        }

    def build_feature_collection(self, features: list[dict]) -> dict:
        """Wrap a list of GeoJSON Features in a FeatureCollection."""
        return {"type": "FeatureCollection", "features": features}
