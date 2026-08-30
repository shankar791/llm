"""
Deterministic Geospatial Analysis & Interpretation Engine for SatQuery AI.
"""
from .raster import GeoTIFFReader, GeoTIFFMetadata, AlignmentChecker
from .processor import GISProcessor
from .geojson import GeoJSONBuilder
from .metrics import area_ha, overlap_iou, change_fraction_to_area
from .projection import reproject_bbox, gdal_geotransform_to_bbox

__all__ = [
    "GeoTIFFReader",
    "GeoTIFFMetadata",
    "AlignmentChecker",
    "GISProcessor",
    "GeoJSONBuilder",
    "area_ha",
    "overlap_iou",
    "change_fraction_to_area",
    "reproject_bbox",
    "gdal_geotransform_to_bbox",
]
