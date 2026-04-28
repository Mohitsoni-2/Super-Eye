# 👁 Super Eye v3.0 — Smart Detection System

Real-time face detection with a modern web UI.  
Upload photos → detect in live webcam or recorded video → log every match with timestamp + confidence.

---

## 📁 Structure

```
Super_Eye/
├── index.html                    ← Open in browser (the UI)
├── static/
│   ├── css/
│   │   ├── style.css             ← All styles
│   │   └── canvas.css            ← Canvas overlay styles
│   └── js/
│       ├── app.js                ← Main app logic + state
│       ├── canvas.js             ← Face box drawing (precise bbox mapping)
│       ├── demo.js               ← Simulated detection (no backend needed)
│       └── ws.js                 ← WebSocket client + frame sender
├── backend/
│   └── server.py                 ← Python WebSocket server (real detection)
├── lost_people_images/           ← Put known person photos here
│   ├── Virat Kohli.png
│   └── ...
├── finding_people_main.py        ← Original standalone script (still works)
├── found_people_report.csv       ← Auto-generated detection log
├── requirements.txt
├── python-3.10.11-amd64.exe      ← Python installer (Windows)
├── cmake-4.3.0-rc2-windows-x86_64.msi
└── vs_BuildTools.exe             ← Required for dlib on Windows
```

---

## 🚀 Quick Start

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** Run the installers in this order first:
> 1. `python-3.10.11-amd64.exe`
> 2. `cmake-4.3.0-rc2-windows-x86_64.msi`
> 3. `vs_BuildTools.exe` (select "Desktop development with C++")
> 4. Then: `pip install dlib` then `pip install face-recognition`

### Step 2 — Run the backend

```bash
python backend/server.py
```

### Step 3 — Open the UI

Open `index.html` in **Chrome or Edge**.  
Top bar shows **BACKEND CONNECTED** in green when live.

---

## 🎮 Using the Interface

| Feature | How |
|---|---|
| Add people | Drag photos onto left panel — name read from filename |
| Live detection | Click **Start Live Detection** → allow camera |
| Recorded video | Switch tab → upload MP4/WebM/MOV |
| Live timestamp | Current date + time (e.g. `23/04/2026  14:32:07`) |
| Video timestamp | Video clock (e.g. `Video: 00:01:34`) |
| Confidence bar | Green ≥85%, Yellow ≥65%, Red <65% |
| Export log | Export CSV button in right panel |
| Settings | ⚙ button — threshold, scan interval, model |

### Keyboard shortcuts
| Key | Action |
|---|---|
| `L` | Switch to live mode |
| `R` | Switch to recorded mode |
| `E` | Export CSV |
| `,` | Open settings |
| `ESC` | Stop detection |

---

## 🧠 Detection Improvements (v3.0)

| Feature | v1 (original) | v3.0 |
|---|---|---|
| Match threshold | 0.6 (dlib default) | **0.52** (fewer false positives) |
| Encoding jitter | 1× | **2× default, up to 5×** |
| Preprocessing | None | **CLAHE + denoising** |
| Model | HOG only | **HOG + CNN option** |
| Bbox scaling | ×4 hardcoded | **Correct ÷SCALE mapping** |
| Canvas overlay | Simple rectangle | **Precise corner brackets + shimmer + fade** |
| Multi-person | ✓ | ✓ improved |
| False positive guard | None | **Cooldown + threshold combo** |

---

## ⚙ Configuration (`backend/server.py`)

```python
MATCH_THRESHOLD = 0.52   # lower = stricter (range: 0.35–0.65)
NUM_JITTERS     = 2      # 1=fast, 2=balanced, 5=highest accuracy
MODEL           = "hog"  # "hog"=CPU fast, "cnn"=GPU best accuracy
SCALE           = 0.25   # frame downscale (0.25=fast, 0.5=more accurate)
SKIP_FRAMES     = 2      # webcam: skip N frames between detections
COOLDOWN_SEC    = 4.0    # min seconds between same-person log entries
```

---

## 🐛 Troubleshooting

**Backend not connecting**  
→ Make sure `python backend/server.py` is running  
→ Check port 8765 is not blocked by firewall  

**"No face found" on photo upload**  
→ Use a clear, front-facing, well-lit photo (min 100×100px)  

**Too many false positives**  
→ Lower `MATCH_THRESHOLD` to 0.45 in `backend/server.py`  

**Too many false negatives (misses real faces)**  
→ Raise `MATCH_THRESHOLD` to 0.58  
→ Increase `NUM_JITTERS` to 5  
→ Switch `MODEL = "cnn"` (needs GPU / CUDA)  

**Bounding box wrong position**  
→ Make sure you're using the latest `backend/server.py` (v3.0) — bbox scaling is fixed  
