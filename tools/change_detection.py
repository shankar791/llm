"""
ChangeDetectionTool — Bi-temporal change detection specialist tool (T4_Change).
Connects to standalone ChangeFormerAdapter supporting both real pretrained inference and mock testing.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional, Union, Literal
import numpy as np
from PIL import Image
from .base import BaseTool, ToolExecutionError
from models.changeformer.adapter import ChangeFormerAdapter


class ChangeDetectionTool(BaseTool):
    """
    Bi-temporal change detection specialist tool wrapping ChangeFormer.
    Evaluates land-cover and structural changes across two temporal acquisitions.
    """
    tool_id = "T4_Change"
    description = "Bi-temporal change detection via ChangeFormer Siamese Transformer"

    def __init__(self, mode: Literal["real", "mock"] = "mock", checkpoint_path: Optional[str] = None):
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self._adapter: Optional[ChangeFormerAdapter] = None

    def _get_adapter(self, mode: str) -> ChangeFormerAdapter:
        if self._adapter is None or self._adapter.mode != mode:
            self._adapter = ChangeFormerAdapter(checkpoint_path=self.checkpoint_path, mode=mode)
            self._adapter.load()
        return self._adapter

    def run(self, image_bytes_t0: Optional[Union[bytes, np.ndarray, Image.Image]] = None,
            image_bytes_t1: Optional[Union[bytes, np.ndarray, Image.Image]] = None,
            mode: Optional[Literal["real", "mock"]] = None,
            modality: str = "optical", **kwargs: Any) -> Dict[str, Any]:
        """
        Execute bi-temporal change detection.

        Args:
            image_bytes_t0: Earlier acquisition image (bytes, numpy array, or PIL Image)
            image_bytes_t1: Later acquisition image (same format/modality)
            mode: 'real' (executes pretrained ChangeFormer) or 'mock' (deterministic fixture)
            modality: Sensor modality (defaults to 'optical')
            kwargs: Additional geospatial metadata (CRS, bounds, transform, dates)

        Returns:
            Dictionary strictly conforming to schemas.models.ToolResult.
        """
        active_mode = mode or kwargs.get("mode") or self.mode
        start_time = time.perf_counter()

        # ======================================================================
        # MOCK MODE BRANCH
        # ======================================================================
        if active_mode == "mock":
            return {
                "tool_id": self.tool_id,
                "answer": "Change analysis detected 14.25 hectares (7.2% of surveyed extent) of change distributed across 14 distinct clusters.",
                "confidence": 0.88,
                "confidence_status": "calibrated_mock",
                "evidence": [
                    {
                        "tool_id": self.tool_id,
                        "label": "change_detected_cluster_1",
                        "coverage_pct": 4.1,
                        "bbox_pixels": [120, 340, 480, 810],
                        "geojson_feature": {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[77.58, 12.97], [77.59, 12.97], [77.59, 12.98], [77.58, 12.98], [77.58, 12.97]]]
                            },
                            "properties": {"cluster_id": 1, "area_ha": 8.1, "change_type": "change_detected"}
                        }
                    },
                    {
                        "tool_id": self.tool_id,
                        "label": "change_detected_cluster_2",
                        "coverage_pct": 3.1,
                        "bbox_pixels": [500, 200, 720, 450],
                        "geojson_feature": {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[77.60, 12.96], [77.61, 12.96], [77.61, 12.97], [77.60, 12.97], [77.60, 12.96]]]
                            },
                            "properties": {"cluster_id": 2, "area_ha": 6.15, "change_type": "change_detected"}
                        }
                    }
                ],
                "evidence_image_b64": None,
                "metadata": {
                    "mock": True,
                    "is_mock": True,
                    "mode": "mock",
                    "status": "success",
                    "change_fraction": 0.072,
                    "total_change_ha": 14.25,
                    "n_clusters": 14,
                    "change_type": "change_detected",
                    "seasonal_filtered": False
                }
            }

        # ======================================================================
        # REAL PRETRAINED MODEL BRANCH
        # ======================================================================
        if image_bytes_t0 is None or image_bytes_t1 is None:
            raise ToolExecutionError(
                "ChangeDetectionTool requires two images (image_bytes_t0 and image_bytes_t1). "
                "One or both were missing."
            )

        try:
            adapter = self._get_adapter(mode="real")
            detect_res = adapter.detect(
                image_t0=image_bytes_t0,
                image_t1=image_bytes_t1,
                metadata=kwargs.get("metadata")
            )
        except Exception as e:
            raise ToolExecutionError(f"ChangeFormer real inference failed: {e}") from e

        meta = detect_res["metadata"]
        change_mask = detect_res["change_mask"]
        change_frac = meta["change_fraction"]
        changed_px = meta["changed_pixels"]
        total_px = meta["total_pixels"]
        elapsed_tool_ms = round((time.perf_counter() - start_time) * 1000, 2)

        answer = (
            f"ChangeFormer bi-temporal analysis ({meta['checkpoint']}) detected {changed_px:,} changed pixels "
            f"({change_frac * 100:.2f}% of surveyed area)."
        )

        evidence = [
            {
                "tool_id": self.tool_id,
                "label": "change_detected",
                "coverage_pct": round(change_frac * 100, 2),
                "bbox_pixels": [0, 0, meta["output_shape"][1], meta["output_shape"][0]],
                "geojson_feature": None  # Populated during deterministic GIS phase
            }
        ]

# Extract and preserve incoming geospatial metadata
        geospatial_meta = {
            "crs": kwargs.get("crs"),
            "transform": kwargs.get("transform"),
            "bounds": kwargs.get("bounds"),
            "resolution": kwargs.get("resolution"),
            "acquisition_dates": kwargs.get("acquisition_dates", {}),
        }
        geospatial_meta = {k: v for k, v in geospatial_meta.items() if v is not None}

        # Auto-extract from GeoTIFF if not explicitly passed
        if ("crs" not in geospatial_meta or "transform" not in geospatial_meta) and isinstance(image_bytes_t0, bytes):
            if image_bytes_t0.startswith(b"II*\x00") or image_bytes_t0.startswith(b"MM\x00*"):
                try:
                    from gis.raster import GeoTIFFReader
                    gtiff_meta = GeoTIFFReader.read_metadata(image_bytes_t0)
                    geospatial_meta.setdefault("crs", gtiff_meta.crs)
                    geospatial_meta.setdefault("transform", gtiff_meta.transform)
                    geospatial_meta.setdefault("bounds", gtiff_meta.bounds)
                    geospatial_meta.setdefault("resolution", gtiff_meta.resolution)
                except Exception:
                    pass

        return {
            "tool_id": self.tool_id,
            "answer": answer,
            "confidence": None,  # Explicitly uncalibrated: model does not provide calibrated probabilities
            "confidence_status": "uncalibrated",
            "evidence": evidence,
            "evidence_image_b64": None,
            "metadata": {
                "mock": False,
                "is_mock": False,
                "mode": "real",
                "status": "success",
                "model": "ChangeFormer",
                "variant": "Official",
                "checkpoint": meta["checkpoint"],
                "change_fraction": change_frac,
                "changed_pixels": changed_px,
                "total_pixels": total_px,
                "input_shape": meta["input_shape"],
                "output_shape": meta["output_shape"],
                "inference_time_ms": meta["inference_time_ms"],
                "total_tool_time_ms": elapsed_tool_ms,
                "change_mask": change_mask,
                "change_type": "change_detected",
                "geospatial": geospatial_meta,
                "domain_limitation": "Pretrained on MineNetCD256; uncalibrated for raw multi-spectral domain shifts."
            }
        }

