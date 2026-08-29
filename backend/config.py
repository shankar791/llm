"""Configuration for SatQuery AI."""
import os

# Model / inference settings (CPU-friendly)
MAX_IMAGE_PIXELS = 4096 * 4096
PATCH_SIZE = 224
CONF_FLOOR = 0.15          # below this, a detected class is ignored in answers

# Where generated evidence images (change masks, grounding boxes) are written
EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Land-cover classes adapted from BigEarthNet-19 taxonomy
BIGEARTHNET_CLASSES = [
    "Urban fabric", "Industrial or commercial units", "Arable land",
    "Permanent crops", "Pastures", "Complex cultivation patterns",
    "Agriculture with natural vegetation", "Agro-forestry areas",
    "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland", "Moors and heathland", "Sclerophyllous vegetation",
    "Transitional woodland/shrub", "Beaches, dunes, sands", "Rocks",
    "Sparsely vegetated areas", "Inland waters", "Marine waters",
    "Coastal wetlands", "Continental wetlands", "Peat bogs",
    "Salt marshes", "Salines", "Intertidal flats",
]
