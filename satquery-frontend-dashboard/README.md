# SatQuery AI — Geospatial Earth Intelligence Dashboard 🛰️✨

A high-performance, liquid-glassmorphic geospatial intelligence interface designed for multitemporal satellite change detection, SAR radar analysis, and spatial reasoning.

![SatQuery Dashboard](https://img.shields.io/badge/SatQuery%20AI-Earth%20Intelligence-3b82f6?style=for-the-badge)
![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900?style=for-the-badge&logo=leaflet)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css)

---

## 🌟 Key Features

### 1. Liquid Glassmorphism UI
- Specular inner reflections, deep frosted refraction, and adaptive opposite background lighting (deep cosmic dark mode background for light glass cards, cool luminous daylight glow for dark glass cards).
- Zero-leak fixed sidebar with quick access to Dashboard, Map Overview, Analyses, Datasets, Projects, Reports, Alerts, and Shared workspaces.
- Seamless dual-icon glass theme toggle (`☀️ / 🌙`).

### 2. Multimodal Spatial Intake Drawer
- **Single Optical Image Mode**: Upload single high-resolution optical rasters (`.jpg`, `.jpeg`, `.png`, `.tif`, `.geotiff`).
- **Multi-Image Pair Mode**: Dedicated dual upload slots for **T1 Baseline** (e.g. May 2023) and **T2 Target** (e.g. May 2025) rasters for bi-temporal ChangeFormer analysis.
- **Sentinel-1 SAR Radar Mode**: Dedicated polarimetric channel slots for **VV Co-Pol Amplitude** and **VH Cross-Pol Amplitude** for all-weather flood and canopy volume assessment.

### 3. Interactive Leaflet GIS Viewport
- Locked high-resolution satellite imagery (Esri World Imagery) centered on **Vignan University (`17.3425° N, 78.7168° E`)**.
- Accurately grounded change polygons for **Vegetation Gain (+18.7%)**, **Built-up Expansion (+12.3%)**, and **Water Retention (-3.6%)**.
- **Interactive Mini-Calendar**: Monthly day grid with one-click orbit epoch switching between May 2023 baseline and May 2025 changes.
- **Magic Wand Tool**: Click two points to draw custom AOI bounding boxes with a floating `[🪄 AOI Active | ✕]` removal badge.
- **Draggable Split-Swipe Comparison**: Centrally constrained bi-temporal swipe slider to inspect pre- vs. post-development changes side-by-side.

### 4. Multi-Session History Switcher
- Instant context switching between 3 example scenarios:
  1. **Vignan University Campus Expansion & Vegetation Survey** (ChangeFormer T4)
  2. **Hussain Sagar Algal Bloom & Water Quality Assessment** (Sentinel-2 Multi-band NDWI)
  3. **East Coast Mangrove Canopy Density Audit** (Sentinel-1 SAR Dual-Pol)
- Fully functional `+ New Chat` workflow with interactive prompt starter chips.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (for local HTTP preview) or any static web server.

### Running Locally
```bash
# Clone the repository
git clone https://github.com/PrakashMB-1213/Sat-Query-Frontend-Dashboard-.git
cd Sat-Query-Frontend-Dashboard-

# Start the local development server
python serve.py 3000
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 📁 Repository Structure
```text
├── index.html                   # Complete self-contained single-page dashboard
├── serve.py                     # Lightweight Python development server
├── interface.md                 # 5-stage roadmap & implementation tracker
├── INTEGRATION_SPEC_AND_ROADMAP.md # LangGraph agent architecture & API specs
└── README.md                    # Project documentation
```

---

## 👤 Author
- **Lead Analyst / Developer**: Prakash ([@PrakashMB-1213](https://github.com/PrakashMB-1213))
