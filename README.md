# 🔍 FIND ME — AI Face Recognition Photo Search

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**Find any person across hundreds of event photos in seconds — using AI face recognition.**

*Built with InsightFace · FAISS · FastAPI · SQLite · Pure HTML/JS*

</div>

---

## ✨ What is FIND ME?

FIND ME is a **local, privacy-first face recognition photo search system**. You upload 1–4 reference photos of a person (or capture them from your webcam), and the system finds every photo of that person in your entire indexed photo collection — even from dark, blurry, far-away, or crowded shots.

Originally built to help find event photos from college fests, it works for any bulk photo collection.

---

## 🚀 Key Features

### 🎯 3-Level Smart Search
Every search automatically runs across **three confidence tiers** — no manual tuning needed:

| Tier | Score Range | Meaning |
|------|-------------|---------|
| 🟢 **Strong Match** | ≥ 0.52 | Near-certain — definitely this person |
| 🟡 **Likely Match** | 0.40 – 0.52 | Probable — most likely this person |
| 🟠 **Possible Match** | 0.30 – 0.40 | Challenging shot — dark / blurry / far away |

> **Smart group-photo filter:** If a photo has more than 10 faces (crowd/auditorium shot), only Strong and Likely matches are shown — Possible matches are filtered out to prevent false positives.

### 📷 Upload OR Capture from Webcam
- **Upload tab** — Drag & drop 1–4 photos, or browse files
- **Camera tab** — Live webcam preview with face guide oval, capture multiple angles (1–4 shots), flash effect on capture, auto-stop on search

### ⚡ Fast Vector Search
- 512-D ArcFace embeddings via **InsightFace** (`buffalo_sc` model)
- Sub-second **FAISS cosine similarity** search across thousands of faces
- Multi-reference photo support — averages embeddings from all reference photos for better accuracy

### 🛡️ Non-Destructive — Original Photos Always Safe
- Original photos in `stock/` are **never touched or modified**
- Photos are cleanly copied to `data/processed_photos/` with sequential numbering (`photo_0001.jpg`, `photo_0002.jpg`, ...)
- Full reverse lookup via SQLite — always know which original file a result came from

### 🌐 Zero-Install Web UI
- Clean dark-mode UI — no external app needed, runs in your browser
- Shows match percentage, tier badge, face count, and direct link to full photo
- Works on both **desktop and mobile**

---

## 🏗️ Architecture

```
FIND ME/
├── stock/                      # ← Put your raw photos here (READ-ONLY)
├── data/
│   ├── processed_photos/       # Numbered copies (photo_0001.jpg ...)
│   ├── metadata.sqlite         # SQLite: photo metadata + face bounding boxes
│   └── faces.index             # FAISS index of 512-D face embeddings
├── src/
│   ├── config.py               # Paths, model name, detection thresholds
│   ├── database.py             # SQLite CRUD operations
│   ├── face_engine.py          # InsightFace model wrapper (singleton)
│   ├── image_analyzer.py       # Image quality analysis (brightness/blur/contrast)
│   ├── indexer.py              # Ingestion pipeline: copy → detect → embed → store
│   ├── searcher.py             # 3-tier FAISS search + group-photo filter
│   └── app.py                  # FastAPI REST API
├── static/
│   └── index.html              # Full web UI (vanilla HTML/CSS/JS + Tailwind CDN)
├── run.py                      # Server entry point
└── requirements.txt            # Python dependencies
```

### Data Flow

```
User uploads/captures photo
        │
        ▼
InsightFace detects face + extracts 512-D embedding
        │
        ▼
Normalize + average (if multiple reference photos)
        │
        ▼
FAISS cosine similarity search (threshold floor: 0.30)
        │
        ▼
3-tier classification → group-photo filter → sorted results
        │
        ▼
JSON response → Web UI renders result cards with tier badges
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/nikhilrawat2005/Find-Me.git
cd Find-Me
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** InsightFace will automatically download the `buffalo_sc` model (~100 MB) on first run.

### 3. Add your photos
Copy your photos into the `stock/` folder:
```
stock/
├── DSC001.JPG
├── DSC002.JPG
├── IMG_5432.jpg
└── ...
```

### 4. Start the server
```bash
# Windows
.\.venv\Scripts\python.exe run.py

# macOS / Linux
.venv/bin/python run.py
```

### 5. Open the web app
Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 📖 How to Use

### Step 1 — Index your photos
Click **"Scan & Index Stock"** in the top-right.  
The system will:
- Copy photos from `stock/` → `data/processed_photos/` with numbered names
- Detect all faces and extract 512-D embeddings
- Store everything in SQLite + FAISS index

*(Only new/unprocessed photos are indexed each time — safe to re-run)*

### Step 2 — Search for a person

**Option A: Upload photos**
1. Click the **Upload Photo** tab
2. Drag & drop 1–4 clear reference photos of the person
3. Click **Search Person**

**Option B: Capture from webcam**
1. Click the **Use Camera** tab
2. Click **Start Camera** → allow browser permission
3. Click **Capture** to take 1–4 shots from different angles
4. Click **Search Person**

### Step 3 — View results
Results appear sorted by confidence score.  
Each card shows:
- **Match percentage** (e.g., `52.3%`)
- **Tier badge** (Strong / Likely / Possible)
- Original filename for traceability
- **View** link to open the full-resolution photo

---

## ⚙️ Configuration

Edit `src/config.py` to change:

```python
INSIGHTFACE_MODEL_NAME = "buffalo_sc"   # Lightweight model (faster)
# Options: "buffalo_l" (more accurate, larger), "buffalo_sc" (fast, compact)

DETECTION_SIZE      = (640, 640)        # Face detection resolution
DETECTION_THRESHOLD = 0.55             # Face detector confidence cutoff
```

Edit tier thresholds in `src/searcher.py`:
```python
TIER_STRONG   = 0.52   # Strong Match floor
TIER_LIKELY   = 0.40   # Likely Match floor
TIER_POSSIBLE = 0.30   # Possible Match floor (minimum)
```

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Face Detection & Embedding | [InsightFace](https://github.com/deepinsight/insightface) (ArcFace `buffalo_sc`) |
| Vector Similarity Search | [FAISS](https://github.com/facebookresearch/faiss) (IndexFlatIP — cosine) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Database | SQLite (via Python `sqlite3`) |
| Frontend | Vanilla HTML/JS + [Tailwind CSS](https://tailwindcss.com/) CDN + [Font Awesome](https://fontawesome.com/) |
| Camera API | Browser `getUserMedia` (WebRTC) |

---

## 📊 Performance

- **Indexing speed:** ~2–5 sec per photo (CPU, depends on face count)
- **Search speed:** < 100ms for 500+ photos (FAISS in-memory)
- **Model size:** `buffalo_sc` — ~20 MB (fast, runs on CPU)
- **Embedding dim:** 512-D normalized float32 vectors

---

## 🗺️ Roadmap

- [ ] Export search results as ZIP of matched photos
- [ ] Batch search (find multiple different people at once)
- [ ] Person tagging / labeling in the UI
- [ ] REST API documentation (Swagger UI at `/docs` already available)
- [ ] Docker container for one-command deployment

---

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <strong>Built with ❤️ by Nikhil Rawat</strong><br>
  <sub>Face recognition · Vector search · Privacy-first · Local-only</sub>
</div>
