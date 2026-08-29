"""
ai.compatibility — Tool–modality compatibility routing for SatQuery AI.

Validates that the tools selected by the intent classifier are compatible
with the modalities present in the uploaded images, and raises a clear error
when the workflow cannot be satisfied.

Modules:
  router — ToolCompatibilityRouter + ToolCompatibilityError
"""
from __future__ import annotations
