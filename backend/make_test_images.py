"""Generate synthetic test satellite images for demo & verification."""
import numpy as np
from PIL import Image
import os

OUT = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

def make_optical_scene(water=0.0, urban=0.0, forest=0.0, size=512, seed=0):
    """Synthetic optical scene with configurable land-cover fractions."""
    r = np.random.default_rng(seed)
    base = np.full((size, size, 3), [120, 140, 90], dtype=np.float32)   # grass baseline
    img = base + r.normal(0, 8, base.shape)

    def fill(mask, color, noise=10):
        m = mask[..., None]
        col = np.array(color, dtype=np.float32)
        img[:] = np.where(m, col + r.normal(0, noise, img.shape), img)

    yy, xx = np.mgrid[0:size, 0:size] / size

    if water > 0:
        cx, cy, rad = 0.30, 0.65, np.sqrt(water / np.pi)
        fill(((xx-cx)**2 + (yy-cy)**2) < rad**2, [40, 70, 130], 6)
    if forest > 0:
        cx, cy, rad = 0.75, 0.30, np.sqrt(forest / np.pi)
        fill(((xx-cx)**2 + (yy-cy)**2) < rad**2, [35, 80, 40], 8)
    if urban > 0:
        cx, cy, rad = 0.70, 0.75, np.sqrt(urban / np.pi)
        m = ((xx-cx)**2 + (yy-cy)**2) < rad**2
        grid = ((xx*size//16 % 2 == 0) | (yy*size//16 % 2 == 0)) & m
        fill(grid, [150, 150, 155], 6)
        fill(m & ~grid, [110, 110, 118], 6)
    return np.clip(img, 0, 255).astype(np.uint8)


def to_sar(arr):
    """Fake SAR conversion: grayscale + speckle."""
    g = arr.mean(axis=-1)
    speckle = rng.gamma(4.0, 0.25, g.shape)
    return np.clip(g * speckle, 0, 255).astype(np.uint8)

# t0 scene: lake + farmland
t0 = make_optical_scene(water=0.08, forest=0.10, seed=1)
Image.fromarray(t0).save(f"{OUT}/optical_t0.png")
# t1 same place but urbanized over the former farmland
t1 = make_optical_scene(water=0.08, urban=0.22, forest=0.06, seed=2)
Image.fromarray(t1).save(f"{OUT}/optical_t1.png")
# SAR pair co-registered to t1
Image.fromarray(to_sar(t1)).save(f"{OUT}/sar_t1.png")
print("written:", os.listdir(OUT))
