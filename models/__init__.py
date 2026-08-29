"""
models — Model adapter layer for SatQuery AI.

Each sub-package contains a thin adapter class that isolates a specific
remote-sensing model from the rest of the codebase. The adapter is the
only file that imports model-specific dependencies.

Sub-packages:
  geochat      — GeoChat VQA, captioning, and grounding adapter
  changeformer — ChangeFormer bi-temporal change detection adapter
  earthgpt     — EarthGPT optical+SAR fusion adapter
  remoteclip   — RemoteCLIP vision-language similarity adapter
"""
from __future__ import annotations
