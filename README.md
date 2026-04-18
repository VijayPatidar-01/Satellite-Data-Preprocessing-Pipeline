# 🛰️ Multi-Band Registration & Geometric Alignment Pipeline

> A **hybrid multi-stage band-to-band registration pipeline** for aligning **Red, Green, and NIR bands** of LISS-4 satellite imagery using **Phase Cross Correlation**, **Edge-Based Cross-Spectral Matching**, **Patch-wise Local Alignment**, **ECC Optimization**, and **Optical Flow** — producing geometrically precise, analysis-ready multi-band GeoTIFFs.

---

## 📌 Project Summary

| Item | Details |
|------|---------|
| **Input** | Raw LISS-4 multi-band imagery (Red, Green, NIR — separate tiles) |
| **Reference Band** | Red (B3) |
| **Target Bands** | Green (B2), NIR (B4) |
| **Alignment Type** | Hybrid — Global + Local + Cross-Spectral |
| **Core Libraries** | Python · OpenCV · scikit-image · NumPy · Rasterio |
| **Output** | Per-tile corrected GeoTIFF with NoData masking |

---

## 🧠 Why a Hybrid Pipeline?

Single-method registration fails on satellite imagery because:

| Problem | Cause | Solution Used |
|---------|-------|---------------|
| Global misalignment | Sensor jitter, orbital offset | Phase Cross Correlation |
| Cross-spectral mismatch | Red ≠ NIR intensity | Edge detection (Canny) + Phase Corr on Edges |
| Local distortions | Terrain, sensor geometry | Patch-wise alignment + Optical Flow |
| Flat region errors | No texture → false shifts | Laplacian variance texture check |
| Over-correction artifacts | Small natural shifts | Shift threshold → skip if below limit |

> This is **not a simple alignment script** — it is a **multi-layer hybrid registration pipeline** with global alignment, local alignment, cross-spectral handling, and artifact prevention built in.

---

## 🔄 Full Pipeline Flowchart

```
╔══════════════════════════════════════════════════════════════╗
║                        INPUT                                 ║
║   RED (reference band)  +  GREEN & NIR (target bands)        ║
╚══════════════════════════════════════════════════════════════╝
                           │
                           ▼
          ┌────────────────────────────────┐
          │    GLOBAL SHIFT DETECTION      │
          │   Phase Cross Correlation      │
          │   → Detect pixel-level offset  │
          │     between bands              │
          └──────────────┬─────────────────┘
                         │
           ┌─────────────┴──────────────┐
           │                            │
           ▼                            ▼
   Shift < Threshold            Shift > Threshold
   (negligible offset)          (significant offset)
           │                            │
           ▼                            ▼
   ✅ Skip Correction           🔧 Apply Alignment
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
                        ▼                               ▼
             ╔═══════════════════╗           ╔═══════════════════╗
             ║  SAME SPECTRAL    ║           ║  CROSS SPECTRAL   ║
             ║  Green ↔ Red      ║           ║  NIR  ↔ Red       ║
             ╚═════════╦═════════╝           ╚═════════╦═════════╝
                       ║                               ║
                       ▼                               ▼
              Coarse Alignment              Edge Detection (Canny)
              (Phase Correlation)           on both Red & NIR
                       │                               │
                       ▼                               ▼
              Patch-wise Alignment          Phase Correlation
              • Divide into patches         on Edge Images
              • Texture check               (intensity-independent)
                (Laplacian variance)                   │
              • Shift magnitude check                  ▼
              • Align valid patches only      Affine Transformation
                       │                     (Translation + Rotation
                       ▼                      + Scaling)
              Optical Flow Refinement                  │
              (Dense pixel displacement)               ▼
              (Handles local distortions)    Skip ECC + Optical Flow
                       │                    (not suitable cross-spec)
                       ▼                               │
              Final Corrected Green                    │
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                            Apply NoData Mask
                            (zero-fill border pixels
                             introduced by warping)
                                       │
                                       ▼
                              Save Output Tiles
                         (green_corrected.tif  /
                           nir_corrected.tif)
```

---

## 🧩 Methods Used — Technical Reference

### 1. 🔵 Intensity-Based Method

| Method | Library | Purpose |
|--------|---------|---------|
| **Phase Cross Correlation** | `skimage.registration` | Global pixel shift detection between bands |

```python
from skimage.registration import phase_cross_correlation

shift, error, diffphase = phase_cross_correlation(
    reference_image = red_band,
    moving_image    = green_band,
    upsample_factor = 10        # sub-pixel precision
)
```

---

### 2. 🟠 Feature-Based (Indirect) — Cross-Spectral

| Method | Library | Purpose |
|--------|---------|---------|
| **Canny Edge Detection** | `cv2.Canny` | Extract structural edges — intensity-independent |
| **Phase Corr on Edges** | `skimage` | Align NIR to Red using edges (not raw DN) |

```python
import cv2

# Extract edges — avoids cross-spectral intensity mismatch
red_edges = cv2.Canny(red_band_uint8, threshold1=50, threshold2=150)
nir_edges = cv2.Canny(nir_band_uint8, threshold1=50, threshold2=150)

# Phase correlation on edge images
shift, _, _ = phase_cross_correlation(red_edges, nir_edges, upsample_factor=10)
```

> **Why edges?** Red and NIR bands have completely different intensity values for the same pixel. Using raw DN would produce false shifts. Edges represent **structural boundaries** which are the same in both bands regardless of spectral response.

---

### 3. 🟡 Patch-wise Area-Based Alignment

| Method | Purpose |
|--------|---------|
| **Image divided into patches** | Handle spatially varying local distortions |
| **Laplacian variance check** | Skip flat/textureless patches (water, cloud) |
| **Shift magnitude check** | Skip patches where shift is below threshold |
| **Per-patch phase correlation** | Local correction for valid patches only |

```python
import numpy as np
import cv2

def laplacian_variance(patch):
    """Texture check — high variance = textured, low = flat."""
    return cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F).var()

def is_valid_patch(patch, texture_threshold=50.0, shift_threshold=0.5):
    texture = laplacian_variance(patch)
    return texture > texture_threshold

# Only correct patches that pass texture & shift checks
# → Prevents noise alignment on flat regions (water, sand, cloud)
```

---

### 4. 🟢 Optical Flow Refinement (Dense)

| Method | Library | Purpose |
|--------|---------|---------|
| **Dense Optical Flow** | `cv2.calcOpticalFlowFarneback` | Sub-pixel dense displacement field |

```python
import cv2

flow = cv2.calcOpticalFlowFarneback(
    prev     = red_band_uint8,
    next     = green_band_uint8,
    flow     = None,
    pyr_scale = 0.5,
    levels   = 3,
    winsize  = 15,
    iterations = 3,
    poly_n   = 5,
    poly_sigma = 1.2,
    flags    = 0
)

# Remap target band using flow field
h, w   = green_band.shape
map_x  = (np.tile(np.arange(w), (h, 1)) + flow[..., 0]).astype(np.float32)
map_y  = (np.tile(np.arange(h), (w, 1)).T + flow[..., 1]).astype(np.float32)
green_corrected = cv2.remap(green_band, map_x, map_y, cv2.INTER_LINEAR)
```

> **Only used for same-spectral alignment (Green ↔ Red).** Not applied for NIR because optical flow depends on intensity similarity.

---

### 5. 🟣 Affine Transformation

| Transform | Handles |
|-----------|---------|
| **Translation** | X/Y pixel offset |
| **Rotation** | Sensor angular misalignment |
| **Scaling** | Minor resolution differences |

```python
import cv2
import numpy as np

def apply_affine_shift(band, shift_y, shift_x):
    """Apply translation via affine warp."""
    h, w = band.shape
    M    = np.float32([[1, 0, shift_x],
                       [0, 1, shift_y]])
    return cv2.warpAffine(band, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)
```

---

### 6. ⚫ NoData Masking

```python
import numpy as np

def apply_nodata_mask(corrected_band, nodata_value=0):
    """
    Zero-fill border pixels introduced by warping.
    Prevents fake values at image edges after shift.
    """
    mask = (corrected_band == nodata_value)
    corrected_band[mask] = 0
    return corrected_band
```

---

## 📊 Alignment Strategy — Decision Table

| Band Pair | Spectral Type | Global Shift | Local Method | Refinement |
|-----------|--------------|--------------|--------------|------------|
| **Green ↔ Red** | Same spectral | Phase Corr | Patch-wise + texture check | Optical Flow |
| **NIR ↔ Red** | Cross spectral | Phase Corr on Edges (Canny) | Affine Transform | None (skip ECC & Flow) |

---

## 🛡️ Quality Control & Artifact Prevention

| Check | Method | Prevents |
|-------|--------|---------|
| **Texture check** | Laplacian variance per patch | Aligning flat regions (water, cloud, sand) |
| **Shift threshold** | Skip if shift < N pixels | Unnecessary correction on already-aligned bands |
| **NoData masking** | Zero-fill border after warp | Fake interpolated values at image edges |
| **Edge-based cross-spectral** | Canny edges not raw DN | False shifts from spectral intensity differences |




---

## 🛠️ Tech Stack

| Library | Version | Role |
|---------|---------|------|
| **OpenCV** | >= 4.5 | Canny, Optical Flow, warpAffine, remap |
| **scikit-image** | >= 0.19 | Phase Cross Correlation |
| **NumPy** | >= 1.21 | Band math, displacement maps |
| **Rasterio** | >= 1.3 | GeoTIFF tile I/O |
| **GDAL** | >= 3.4 | Geospatial metadata, projection |

---

## 📚 References

- [scikit-image — Phase Cross Correlation](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation)
- [OpenCV — Optical Flow (Farneback)](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
- [OpenCV — Canny Edge Detection](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html)
- Guizar-Sicairos, M. et al. (2008). *Efficient subpixel image registration algorithms.* Optics Letters, 33(2), 156–158.
- Farneback, G. (2003). *Two-frame motion estimation based on polynomial expansion.* SCIA 2003, LNCS 2749, 363–370.

---

## 👤 Author

**VIJAY PATIDAR**
- 🔗 GitHub: [@Vijay patidar](https://github.com/VijayPatidar-01)
- 📧 Email: patidr.vijay9973@gmail.com



> *Built with 🖼️ OpenCV · 📐 Phase Correlation · 🌊 Optical Flow · 🛰️ Rasterio · 🧩 Hybrid Registration*
