# 🔍 FIND ME — AI Face Recognition Photo Search & Event Manager

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple?style=flat-square)
![Google Drive API](https://img.shields.io/badge/Google%20Drive-Integrated-yellow?style=flat-square&logo=googledrive)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**Find any person across hundreds or thousands of event photos in seconds — using AI face recognition with direct Google Drive integration.**

*Built with InsightFace · FAISS · FastAPI · Google Drive API · SQLite · Pure Tailwind & JS UI*

</div>

---

## ✨ What is FIND ME?

FIND ME is an **AI-powered face recognition search and event photo management system**. You upload or capture reference photos of a person via your webcam, and the system instantly identifies all photos containing that person across local collections or entire Google Drive event folders — even in challenging lighting, distant shots, group crowds, or varying angles.

---

## 🚀 Key Features

### ☁️ Autonomous Google Drive Event Pipeline
- **1-Click Cloud Sync:** Paste any Google Drive folder link to automatically ingest an event.
- **Permission Pre-flight Verification:** Validates Google Service Account "Editor" permissions before performing operations, preventing silent failures.
- **Auto Duplicate Cleanup:** Detects and trashes duplicate uploads directly on Drive based on file hash and names.
- **Direct Drive Renaming & Sequential Indexing:** Auto-renames Drive images cleanly to sequential formats (`photo_0001.jpg`, `photo_0002.jpg`, ...).
- **In-Memory Streaming (Zero Local Disk Waste):** Streams photos into RAM for embedding generation and passes them directly to FAISS without cluttering local storage.

### 🔄 Event Management (Re-Sync & Delete)
- **Per-Event Re-Sync:** Re-check any connected Google Drive folder with 1 click. Verifies permissions, purges newly added duplicates, renames new files sequentially, and indexes unindexed photos.
- **Event Deletion:** Cleanly delete events and their associated face embeddings from SQLite & FAISS right from the dropdown selector.

### ⚡ High-Performance Multi-Threaded Engine
- **Concurrent Prefetching Pipeline:** Uses thread-isolated background workers (`ThreadPoolExecutor` + `queue.Queue`) to decouple network downloads from CPU face analysis.
- **Smart Image Downscaling:** Downscales oversized DSLR images (down to 1600px max edge) before passing into ArcFace, cutting CPU inference time from ~5s down to ~0.5s per photo with 0 loss in accuracy.
- **Thread-Local SSL Architecture:** Zero SSL drops, bad record MAC, or OpenSSL race conditions.
- **Local Recruitment / Event Folders:** Automatically processes event subfolders inside `local/` (e.g. `local/local_recruitment/`).

### 🎯 Dynamic Scale-Aware & Crowd-Aware Search Precision
Every search automatically classifies matches with strict precision rules:

| Tier | Score Range | Condition & Behavior |
|------|-------------|-----------------------|
| 🟢 **Strong Match** | ≥ 0.52 (52%+) | Near-certain match — guaranteed exact person |
| 🟡 **Likely Match** | 0.40 – 0.52 (40%–52%) | High confidence match across lighting and angles |
| 🟠 **Possible / Distant Match** | 0.30 – 0.40 (30%–40%) | **Dynamic Scale Rule:** *Strictly restricted to small/distant faces (<120px bounding box)*. Normal/clear faces must meet ≥ 40% similarity to eliminate false positives |

> **Smart Crowd & Group Filter:** For group photos containing >10 faces, loose possible matches are automatically rejected to eliminate false alarms in large audiences.

### 🎨 Deep Black & Electric Blue Cyber UI
- Sleek dark theme with **Plus Jakarta Sans** typography and neon cyber blue accents.
- Distinct color coding & badges for **Local Events** (`💻 [LOCAL]`) vs **Google Drive Events** (`☁️ [DRIVE]`).
- Dynamic header counters automatically display the exact photo and face counts scoped to whichever event is currently selected.

### 📷 Dual Input Modes (Upload & Auto-Camera)
- **Upload Mode:** Drag-and-drop 1–4 clear reference images.
- **Webcam Mode:** Live camera stream with visual oval alignment guide and auto-capture on steady face detection.

### 📦 ZIP Export & Report
- Download all matched photos as a bundled ZIP file.
- Export detailed match reports in JSON.

---

## 🏗️ Architecture & Project Structure

```
FIND ME/
├── stock/                      # Local raw photos (optional fallback)
├── data/
│   ├── metadata.sqlite         # SQLite database: events, photos, face bounding boxes
│   ├── faces.index             # FAISS index storing 512-D ArcFace vectors
│   └── processed_photos/       # Processed local stock images
├── src/
│   ├── config.py               # Application configurations, paths, and thresholds
│   ├── database.py             # SQLite CRUD, Event lifecycle, deletion, and stats
│   ├── face_engine.py          # InsightFace ArcFace model wrapper (singleton)
│   ├── gdrive_service.py       # Google Drive API client (auth, stream, rename, trash)
│   ├── image_analyzer.py       # Image quality and lighting validator
│   ├── indexer.py              # Ingestion engine: Drive sync, duplicate trashing, FAISS indexing
│   ├── searcher.py             # FAISS similarity search & multi-tier categorization
│   └── app.py                  # FastAPI REST API endpoints
├── static/
│   └── index.html              # Responsive web UI (Tailwind CSS, FontAwesome)
├── credentials.json            # Google Service Account credentials (optional for Drive)
├── run.py                      # Server runner
└── requirements.txt            # Python dependencies
```

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/nikhilrawat2005/Find-Me.git
cd Find-Me
```

### 2. Set up Python environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Google Drive Integration Setup (Optional for Drive Folders)
1. Place your Google Cloud Service Account key file as `credentials.json` in the root folder.
2. Share your Google Drive event folders with the Service Account email (`client_email` in `credentials.json`) with the **"Editor"** role.

### 4. Start the Application
```bash
python run.py
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## 📖 How to Use

1. **Add an Event:**
   - Click **"Add / Sync Event"** in the top bar.
   - Enter your Google Drive folder link or folder ID.
   - The system checks permissions, cleans duplicates, renames images (`photo_0001.jpg`...), and indexes faces directly in RAM.
2. **Re-Sync or Delete:**
   - Select an event from the top **Event** dropdown.
   - Click the **Re-Sync (<i class="fa-solid fa-rotate"></i>)** button to check for new files or re-run duplicate/naming cleanups.
   - Click the **Delete (<i class="fa-solid fa-trash-can"></i>)** button to remove the event data from the database.
3. **Search:**
   - Upload or capture 1–4 photos of the target person.
   - Click **Search Person** to view matched photos categorized by confidence level.

---

## 📄 License

MIT License — free to use and build upon.

