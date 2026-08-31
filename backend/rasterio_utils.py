"""Raster I/O, metadata extraction, and modality detection for satellite imagery."""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import array_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ---------------------------------------------------------------- sensor detection
SENSOR_PATTERNS = [
    (r"sentinel[-_]?2|s2[ab]?", "Sentinel-2"),
    (r"sentinel[-_]?1|s1[ab]?", "Sentinel-1"),
    (r"landsat[-_]?8|landsat[-_]?9|l8|l9", "Landsat-8/9"),
    (r"landsat[-_]?[457]", "Landsat-4/5/7"),
    (r"cartosat[-_]?2[s]?", "Cartosat-2S"),
    (r"cartosat[-_]?3", "Cartosat-3"),
    (r"resourcesat[-_]?2|resourcesat[-_]?2a", "Resourcesat-2/2A"),
    (r"risat[-_]?1|risat[-_]?2b", "RISAT-1/2B"),
    (r"worldview[-_]?[23]", "WorldView-2/3"),
    (r"planet[-_]?scope|dove|sky[sat]?", "PlanetScope"),
    (r"modis", "MODIS"),
    (r"viirs", "VIIRS"),
]

DATE_PATTERNS = [
    r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})",          # YYYY-MM-DD or YYYYMMDD
    r"(\d{8})",                                    # YYYYMMDD
    r"(\d{4})(\d{2})(\d{2})",                      # YYYYMMDD no sep
]

def _guess_sensor(filename: str) -> str:
    name = filename.lower()
    for pattern, sensor in SENSOR_PATTERNS:
        if re.search(pattern, name):
            return sensor
    return "Unknown"

def _guess_date(filename: str, tags: dict) -> Optional[str]:
    # 1. Try GDAL/rasterio tags first
    for key in ("TIFFTAG_DATETIME", "datetime", "acquisition_date", "date"):
        if key in tags:
            val = tags[key]
            if isinstance(val, str) and val.strip():
                return _normalize_date(val)
    # 2. Try EXIF via PIL (if we open with PIL)
    # 3. Fallback: parse from filename
    name = filename.lower()
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, name)
        if m:
            groups = m.groups()
            if len(groups) == 3:
                return f"{groups[0]}-{groups[1]}-{groups[2]}"
            elif len(groups) == 1 and len(groups[0]) == 8:
                s = groups[0]
                return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return None

def _normalize_date(s: str) -> Optional[str]:
    """Normalize various date formats to YYYY-MM-DD."""
    s = s.strip()
    # ISO-like
    m = re.match(r"^(\d{4})[-/:]?(\d{2})[-/:]?(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # EXIF format: "2026:06:11 10:30:00"
    m = re.match(r"^(\d{4}):(\d{2}):(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


# ---------------------------------------------------------------- RasterInput
class RasterInput:
    """Normalized wrapper around an uploaded satellite image with full metadata."""

    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.data = data
        self.arr = self._decode()
        self.modality = self._detect_modality()
        self.metadata = self._extract_metadata()

    def _decode(self) -> np.ndarray:
        """Decode image to numpy array (PIL for non-GeoTIFF, rasterio for GeoTIFF)."""
        fmt = self.filename.rsplit(".", 1)[-1].lower()
        if fmt in ("tif", "tiff") and HAS_RASTERIO:
            try:
                with rasterio.open(io.BytesIO(self.data)) as src:
                    # Read all bands
                    arr = src.read()  # shape: (bands, H, W)
                    if arr.shape[0] == 1:
                        return arr[0]  # (H, W)
                    return np.transpose(arr, (1, 2, 0))  # (H, W, bands)
            except Exception:
                pass  # fall through to PIL
        # PIL fallback (PNG, JPEG, etc.)
        img = Image.open(io.BytesIO(self.data))
        if img.mode not in ("RGB", "L", "RGBA"):
            img = img.convert("RGB" if img.mode in ("RGBA", "P") else img.mode)
        return np.array(img)

    def _detect_modality(self) -> str:
        """
        Heuristic SAR vs optical detection:
         - SAR: single-band grayscale with speckle, or filename hints
         - Optical: 3+ band color or multispectral
        """
        name = self.filename.lower()
        if any(k in name for k in ("sar", "_s1", "s1a", "s1b", "risat")):
            return "sar"
        if self.arr.ndim == 2 or (self.arr.ndim == 3 and self.arr.shape[2] == 1):
            return "sar"
        return "optical"

    def _extract_metadata(self) -> dict:
        """Extract comprehensive metadata from file."""
        fmt = self.filename.rsplit(".", 1)[-1].lower()
        meta = {
            "filename": self.filename,
            "size_bytes": len(self.data),
            "width": self.arr.shape[1],
            "height": self.arr.shape[0],
            "bands": self.arr.shape[2] if self.arr.ndim == 3 else 1,
            "format": fmt,
            "modality": self.modality,
        }

        # Try rasterio for GeoTIFF metadata
        if fmt in ("tif", "tiff") and HAS_RASTERIO:
            try:
                with rasterio.open(io.BytesIO(self.data)) as src:
                    meta.update(self._extract_rasterio_metadata(src))
            except Exception:
                pass  # keep PIL-derived metadata

        # Sensor & date (from filename/tags as fallback)
        meta["sensor"] = meta.get("sensor") or _guess_sensor(self.filename)
        meta["acquisition_date"] = meta.get("acquisition_date") or _guess_date(self.filename, meta.get("tags", {}))
        meta["crs"] = meta.get("crs") or "EPSG:4326"  # default assumption
        meta["resolution_m"] = meta.get("resolution_m") or self._estimate_resolution(meta)

        return meta

    def _extract_rasterio_metadata(self, src) -> dict:
        """Extract metadata from rasterio dataset."""
        meta = {}

        # CRS
        if src.crs:
            meta["crs"] = src.crs.to_string()
            meta["crs_epsg"] = src.crs.to_epsg()

        # Bounds (in CRS units)
        if src.bounds:
            meta["bounds"] = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            meta["bounds_lonlat"] = self._reproject_bounds_to_lonlat(src.bounds, src.crs)

        # Transform & resolution
        if src.transform:
            meta["transform"] = list(src.transform)[:6]
            # Resolution in CRS units (meters for UTM, degrees for geographic)
            meta["resolution_x"] = abs(src.transform.a)
            meta["resolution_y"] = abs(src.transform.e)
            meta["resolution_m"] = self._estimate_resolution_from_transform(src.transform, src.crs)

        # Tags
        tags = dict(src.tags())
        if tags:
            meta["tags"] = tags
            # Sensor from tags
            for k in ("sensor", "platform", "satellite", "instrument"):
                if k in tags:
                    meta["sensor"] = tags[k]
                    break
            # Acquisition date from tags
            for k in ("datetime", "acquisition_date", "date", "time"):
                if k in tags:
                    norm = _normalize_date(str(tags[k]))
                    if norm:
                        meta["acquisition_date"] = norm
                        break

        # Band descriptions
        if src.descriptions:
            meta["band_descriptions"] = [d for d in src.descriptions if d]

        return meta

    def _reproject_bounds_to_lonlat(self, bounds, crs) -> Optional[list]:
        """Reproject bounds to EPSG:4326 (lon/lat)."""
        if not HAS_RASTERIO:
            return None
        try:
            from rasterio.warp import transform_bounds
            if crs and crs.to_string() != "EPSG:4326":
                b = transform_bounds(crs, "EPSG:4326", *bounds)
                return [b[0], b[1], b[2], b[3]]
        except Exception:
            pass
        return [bounds.left, bounds.bottom, bounds.right, bounds.top]

    def _estimate_resolution_from_transform(self, transform, crs) -> float:
        """Estimate ground resolution in meters."""
        if not crs:
            return 10.0  # default guess
        try:
            crs_str = crs.to_string()
            if "EPSG:4326" in crs_str or "WGS84" in crs_str:
                # Degrees → meters at equator (rough)
                deg_res = min(abs(transform.a), abs(transform.e))
                return deg_res * 111320  # meters per degree
            else:
                # Projected CRS — transform units are typically meters
                return min(abs(transform.a), abs(transform.e))
        except Exception:
            return 10.0

    def _estimate_resolution(self, meta: dict) -> float:
        """Fallback resolution estimate."""
        if "resolution_m" in meta:
            return meta["resolution_m"]
        if "resolution_x" in meta and "resolution_y" in meta:
            return min(meta["resolution_x"], meta["resolution_y"])
        # Heuristic by sensor
        sensor = meta.get("sensor", "").lower()
        if "sentinel-2" in sensor:
            return 10.0
        if "sentinel-1" in sensor:
            return 10.0
        if "landsat" in sensor:
            return 30.0
        if "cartosat-2" in sensor:
            return 2.0
        if "cartosat-3" in sensor:
            return 0.5
        if "risat" in sensor:
            return 5.0
        return 10.0

    def thumbnail(self, max_side: int = 768) -> np.ndarray:
        """Resize preserving aspect ratio for model input."""
        h, w = self.arr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = Image.fromarray(
                self.arr if self.arr.dtype == np.uint8 else self.arr.astype(np.uint8)
            )
            img = img.resize((int(w * scale), int(h * scale)))
            return np.array(img)
        return self.arr

    # ------------------------------------------------------------ convenience
    @property
    def bounds_lonlat(self) -> Optional[list]:
        """Returns [min_lon, min_lat, max_lon, max_lat] in EPSG:4326."""
        return self.metadata.get("bounds_lonlat")

    @property
    def center_lonlat(self) -> Optional[tuple]:
        """Returns (center_lon, center_lat) in EPSG:4326."""
        b = self.bounds_lonlat
        if b:
            return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        return None


def validate_inputs(rasters: list[RasterInput]) -> dict:
    """Check count/modality/format compatibility; returns a scenario tag."""
    n = len(rasters)
    if n == 0:
        raise ValueError("No images uploaded.")
    if n > 4:
        raise ValueError("Maximum 4 images per query.")

    formats = {r.metadata["format"] for r in rasters}
    if not formats <= {"tif", "tiff", "png", "jpg", "jpeg", "webp", "bmp"}:
        raise ValueError(f"Unsupported format(s): {formats}. Use GeoTIFF/TIFF/PNG/JPEG/WEBP.")

    modalities = [r.modality for r in rasters]
    sizes = [(r.metadata["height"], r.metadata["width"]) for r in rasters]

    scenario = {
        "count": n,
        "modalities": modalities,
        "sizes": sizes,
    }
    if n == 1:
        scenario["scenario"] = "single_image"
    elif n == 2:
        same_modality = modalities[0] == modalities[1]
        similar_size = abs(sizes[0][0] - sizes[1][0]) < 0.25 * max(sizes[0]) and \
                       abs(sizes[0][1] - sizes[1][1]) < 0.25 * max(sizes[1])
        if same_modality and similar_size:
            scenario["scenario"] = "bi_temporal_pair"
        elif not same_modality:
            scenario["scenario"] = "cross_modal_pair"
        else:
            scenario["scenario"] = "two_independent"
    else:
        scenario["scenario"] = f"multi_{n}"

    return scenario
