"""
Raster loading, metadata extraction, channel policy, and alignment validation for SatQuery AI.
Uses Rasterio for deterministic geospatial raster I/O.
"""
from __future__ import annotations
import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


class GeoTIFFMetadata:
    """Structured container for extracted raster geospatial metadata."""
    def __init__(
        self,
        width: int,
        height: int,
        crs: str,
        transform: List[float],
        bounds: List[float],
        resolution: Tuple[float, float],
        band_count: int,
        nodata: Optional[float] = None,
        driver: str = "GTiff",
        dtype: str = "uint8"
    ):
        self.width = width
        self.height = height
        self.crs = crs
        self.transform = transform
        self.bounds = bounds
        self.resolution = resolution
        self.band_count = band_count
        self.nodata = nodata
        self.driver = driver
        self.dtype = dtype

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": self.transform,
            "bounds": self.bounds,
            "resolution": list(self.resolution),
            "band_count": self.band_count,
            "nodata": self.nodata,
            "driver": self.driver,
            "dtype": self.dtype,
        }


class GeoTIFFReader:
    """
    Deterministic GeoTIFF parser and preprocessor.
    Extracts authentic geospatial headers and enforces an explicit RGB channel policy.
    """

    @staticmethod
    def read_metadata(source: Union[str, bytes]) -> GeoTIFFMetadata:
        """Extract metadata from GeoTIFF file path or memory bytes."""
        if isinstance(source, bytes):
            with rasterio.open(io.BytesIO(source)) as src:
                return GeoTIFFReader._extract_meta(src)
        elif isinstance(source, str):
            with rasterio.open(source) as src:
                return GeoTIFFReader._extract_meta(src)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    @staticmethod
    def _extract_meta(src: rasterio.DatasetReader) -> GeoTIFFMetadata:
        crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
        t = src.transform
        transform_list = [t.a, t.b, t.c, t.d, t.e, t.f]
        b = src.bounds
        bounds_list = [b.left, b.bottom, b.right, b.top]
        res = (abs(t.a), abs(t.e))

        return GeoTIFFMetadata(
            width=src.width,
            height=src.height,
            crs=crs_str,
            transform=transform_list,
            bounds=bounds_list,
            resolution=res,
            band_count=src.count,
            nodata=src.nodata,
            driver=src.driver,
            dtype=str(src.dtypes[0])
        )

    @staticmethod
    def read_rgb(
        source: Union[str, bytes],
        band_mapping: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, GeoTIFFMetadata]:
        """
        Extract 3-channel RGB numpy array (H, W, 3) in uint8 [0, 255] and metadata.

        Channel Policy:
        - 3 bands: Assumed visual RGB -> bands [1, 2, 3] (Red, Green, Blue)
        - 4 bands (RGB+NIR): Selects visual bands [1, 2, 3]
        - Sentinel-2 (>=12 bands): Explicitly selects visual bands [4, 3, 2] (B4=Red, B3=Green, B2=Blue)
        - 1 band (Grayscale/SAR): Duplicates single band into 3 channels [1, 1, 1]
        - Custom: Uses user-supplied 1-based band_mapping list of 3 integers

        Applies 2%-98% cumulative percentile stretching for 16-bit / float rasters.
        """
        def _process(src: rasterio.DatasetReader) -> Tuple[np.ndarray, GeoTIFFMetadata]:
            meta = GeoTIFFReader._extract_meta(src)
            count = src.count

            # Determine bands to read
            if band_mapping is not None:
                if len(band_mapping) != 3:
                    raise ValueError(f"band_mapping must contain exactly 3 band indices, got {band_mapping}")
                bands_to_read = band_mapping
            elif count == 1:
                bands_to_read = [1, 1, 1]
            elif count == 3:
                bands_to_read = [1, 2, 3]
            elif count == 4:
                bands_to_read = [1, 2, 3]
            elif count >= 12:
                # Standard Sentinel-2 L1C/L2A layout: B4 (Red)=4, B3 (Green)=3, B2 (Blue)=2
                bands_to_read = [4, 3, 2]
            else:
                raise ValueError(
                    f"Unsupported GeoTIFF band count ({count}). "
                    f"Please provide an explicit 3-band mapping (e.g. band_mapping=[1, 2, 3])."
                )

            # Read selected bands
            raw_bands = [src.read(b) for b in bands_to_read]
            stack = np.stack(raw_bands, axis=-1)  # (H, W, 3)

            # Normalization / Contrast Stretch to uint8 [0, 255]
            if stack.dtype == np.uint8:
                rgb = stack
            else:
                # 16-bit uint16 or float: percentile 2%-98% stretch
                rgb = np.zeros(stack.shape, dtype=np.uint8)
                for c in range(3):
                    chan = stack[:, :, c].astype(np.float32)
                    valid_mask = np.isfinite(chan)
                    if meta.nodata is not None:
                        valid_mask &= (chan != meta.nodata)
                    if np.any(valid_mask):
                        p2, p98 = np.percentile(chan[valid_mask], (2, 98))
                        if p98 > p2:
                            stretched = (chan - p2) / (p98 - p2)
                            rgb[:, :, c] = (np.clip(stretched, 0.0, 1.0) * 255).astype(np.uint8)
                        else:
                            rgb[:, :, c] = np.clip(chan, 0, 255).astype(np.uint8)

            return rgb, meta

        if isinstance(source, bytes):
            with rasterio.open(io.BytesIO(source)) as src:
                return _process(src)
        elif isinstance(source, str):
            with rasterio.open(source) as src:
                return _process(src)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")


class AlignmentChecker:
    """Verifies geospatial alignment and temporal compatibility between two GeoTIFFs."""

    @staticmethod
    def verify_alignment(meta_t0: GeoTIFFMetadata, meta_t1: GeoTIFFMetadata) -> Tuple[bool, List[str]]:
        """
        Check whether two rasters share compatible CRS, matching spatial extent, and resolution.
        """
        errors = []

        # 1. CRS Check
        if meta_t0.crs != meta_t1.crs:
            errors.append(f"CRS mismatch: T0 has {meta_t0.crs}, T1 has {meta_t1.crs}.")

        # 2. Dimension Check
        if meta_t0.width != meta_t1.width or meta_t0.height != meta_t1.height:
            errors.append(
                f"Dimension mismatch: T0 is ({meta_t0.width}x{meta_t0.height}), "
                f"T1 is ({meta_t1.width}x{meta_t1.height})."
            )

        # 3. Spatial Bounds Overlap Check
        b0 = meta_t0.bounds
        b1 = meta_t1.bounds
        overlap_x = max(0.0, min(b0[2], b1[2]) - max(b0[0], b1[0]))
        overlap_y = max(0.0, min(b0[3], b1[3]) - max(b0[1], b1[1]))
        if overlap_x <= 0 or overlap_y <= 0:
            errors.append(f"Zero spatial overlap between T0 bounds {b0} and T1 bounds {b1}.")

        return len(errors) == 0, errors
