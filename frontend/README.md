# EDOLUS — 3D Interactive Web Experience Clone

A standalone clone of the interactive 3D WebGL web experience from [https://edolus.com/](https://edolus.com/).

---

## 🌟 Overview

**EDOLUS** is a WebGL2 3D interactive application built with the PlayCanvas engine. It delivers cinematic animations, camera walks, real-time lighting, interactive 3D orbital satellites and compute hardware models, animated shaders, and multi-track audio.

---

## 🚀 Quick Start

### 1. Run with Python (Recommended)

Run the included multithreaded server with full byte-range media streaming support:

```bash
python serve.py 3000
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser.

### 2. Run with Node.js / `npx serve`

```bash
npx serve -l 3000 .
```

### 3. Run with VS Code Live Server

Open the repository in VS Code and click **Go Live** on `index.html`.

> **Note**: A local web server is required because modern browsers restrict WebGL shaders, WebAssembly modules, and local `fetch()` calls over the `file://` protocol.

---

## 📦 What's Included

- **3D Models (`.glb`)**:
  - `Tesla-ready2.glb` (7.3 MB)
  - `ComputeTrayJOINED.glb` (7.5 MB)
  - `StarlinkV2-1.glb` (3.6 MB)
  - `GDX-3PODS-F2.glb` (1.5 MB)
  - `DiamondV4.glb` (1.4 MB)
  - `AutonomousDeployment.glb`
  - `MAP.glb`, `processor.glb`, `CurvedScreen.glb`, `triangle3D.glb`
- **Audio Stems & Sound FX (`.ogg`)**:
  - Multi-track stems: `melody_stem`, `instruments_stem`, `bass_stem`
  - Interactive UI sound effects: `BTNclick2`, `diamond2`, `map`, `rack`, `computecore2`, `textchip2`, `quantum5`, `UIscreen`, `AICHIP`, `intelligent`, `Scrambletext`
- **HD Video Backgrounds (`.mp4`)**:
  - `Space-compressNOAUDIO-YTfullHD.mp4` (6.0 MB)
  - `Seedance2-0_r2v_00002YTHD_lowbitrate.mp4` (5.7 MB)
- **Universal Textures & Shaders**:
  - WebAssembly Basis Universal compressed textures (`.basis`)
  - Normal maps, ambient occlusion, diffuse, reflection maps, and skyboxes
- **Engine Runtime**:
  - PlayCanvas Engine (`playcanvas-stable.min.js`)
  - Basis Transcoder WASM decoder (`basis.wasm.js`, `basis.wasm.wasm`, `basis.js`)
  - GSAP (`gsap.min.js`) animation library
  - Core game logic (`__game-scripts.js`), loaders, scene graph (`2509662.json`), and asset registry (`config.json`)

---

## 🛠️ Project Structure

```
├── index.html                  # Main application entrypoint
├── styles.css                  # Core CSS and UI styling
├── gsap.min.js                 # GSAP animation runtime
├── playcanvas-stable.min.js    # PlayCanvas WebGL runtime
├── __settings__.js             # Engine initialization settings
├── __modules__.js              # Module & WASM loader
├── __start__.js                # Application bootstrapper
├── __loading__.js              # Stepped odometer loader & splash sequence
├── __game-scripts.js           # Interactive application scripts & shaders
├── config.json                 # PlayCanvas asset registry & scene configs
├── 2509662.json                # Main scene entity hierarchy
├── serve.py                    # Multithreaded local development server
├── files/
│   └── assets/                 # 3D models, textures, audio, and videos
└── README.md                   # Documentation
```

---

## ⚡ Tech Stack

- **Graphics**: WebGL2 / PlayCanvas Engine
- **3D Formats**: Binary glTF (`.glb`), Basis Universal Texture Compression (`.basis`)
- **Animation**: GSAP + Custom Web Worker Offscreen Canvas renderers
- **Audio**: Multi-channel HTML5 Web Audio API
- **WebAssembly**: Google Basis Universal Transcoder WASM

---

## 📄 License & Attribution

All 3D assets, trademarks, and design rights belong to **EDOLUS** ([edolus.com](https://edolus.com/)). This repository is cloned for development and showcase purposes.
