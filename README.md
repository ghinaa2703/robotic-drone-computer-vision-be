# 🚀 Backend Drone Vision System (FastAPI)

Ini adalah `backend` aplikasi `Drone Vision System`, yang bertanggung jawab untuk memproses `real-time video stream` dari drone (atau simulasi `RTMP stream`) menggunakan algoritma `computer vision` seperti deteksi objek YOLO dan pengenalan wajah FaceNet, serta mengirimkan hasil pemrosesan dan data analitik ke `frontend web dashboard`.

## ✨ Cara Menjalankan (Get Started)

Ikuti langkah-langkah di bawah ini untuk menjalankan `backend` dan menyiapkan `environment` yang diperlukan.

### 1. Prasyarat

Pastikan Anda telah menginstal yang berikut ini di sistem Anda:

*   **Python 3.9+** (Disarankan Python 3.10 atau 3.11)
*   **pip** (Pengelola `package` Python, biasanya sudah termasuk dengan instalasi Python)
*   **Docker Desktop** (untuk menjalankan `RTMP server` Nginx)
*   **OBS Studio** (untuk mensimulasikan `stream` drone)
*   **FFmpeg** (opsional, untuk menguji `stream RTMP` dengan VLC atau `host stream` lokal secara manual)

### 2. Kloning Repositori

```bash
git clone <url_repo>
cd drone-vision-be
```

### 3. Buat dan Aktifkan `Virtual Environment`

Sangat disarankan untuk menggunakan `virtual environment` untuk mengelola `dependency` proyek.

```bash
python -m venv env
# Aktifkan virtual environment:
# Di Windows: .\env\Scripts\activate
# Di Linux/macOS: source env/bin/activate
```

### 4. Instal `Dependency` Python

Setelah `virtual environment` aktif, instal semua `library` yang diperlukan:

```bash
pip install -r requirements.txt
```
*(Pastikan `requirements.txt` Anda berisi `fastapi`, `uvicorn`, `websockets`, `opencv-python-headless`, `numpy`, `face-recognition`, `ultralytics`, `torch`, `torchvision`, `scipy`, `pickle-mixin` jika digunakan.)*

**Catatan Penting untuk `CUDA` (GPU Acceleration):**
Jika Anda memiliki `GPU` (`NVIDIA`) dan ingin akselerasi performa `deep learning` (YOLO dan `face_recognition`), instal `PyTorch` dengan dukungan `CUDA` **setelah** menginstal `requirements.txt` dan **sebelum** menjalankan `backend`.

Kunjungi `https://pytorch.org/get-started/locally/` dan ikuti instruksi untuk versi `CUDA` yang sesuai dengan `driver` `GPU` Anda. Contoh:
```bash
# Contoh untuk PyTorch 2.3.0 dengan CUDA 12.1
pip uninstall torch torchvision torchaudio # Hapus versi CPU jika sudah terinstal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Pastikan `cv2.getBuildInformation()` menunjukkan `FFMPEG: YES` dan `CUDA: YES` setelah instalasi.

### 5. Siapkan `RTMP Stream Server` (menggunakan Docker)

`Backend` ini dirancang untuk menerima `stream RTMP` (misalnya dari OBS Studio). Cara termudah adalah menjalankan `Nginx-RTMP server` menggunakan Docker.

```bash
docker run -d \
  -p 1935:1935 \
  -p 8080:80 \
  --name nginx-rtmp-server \
  tiangolo/nginx-rtmp:latest
```
Verifikasi bahwa `server` berjalan dengan mengunjungi `http://localhost:8080` atau `http://localhost:8080/rtmp-status` di `browser` Anda.

### 6. Siapkan `Dataset` Wajah (untuk `Face Recognition`)

*   Buat `folder` `dataset/face_recognition` di `root directory` `backend` Anda:
    ```
    drone-vision-be/
    ├── app/
    ├── dataset/
    │   └── face_recognition/  <-- BUAT FOLDER INI
    │       ├── NamaOrang1/
    │       │   ├── img1.jpg
    │       │   └── img2.jpg
    │       └── NamaOrang2/
    │           ├── imgA.png
    │           └── imgB.png
    └── yolov8n.pt
    └── ... (main.py, etc.)
    ```
*   Letakkan gambar-gambar wajah dari orang yang ingin Anda kenali di dalam subfolder dengan nama orang tersebut.

### 7. Muat Model YOLO (jika belum ada `yolov8n.pt`)

Pastikan `file` model YOLO yang akan digunakan (misalnya `yolov8n.pt`) ada di `root directory` `drone-vision-be/`. Jika tidak, Anda bisa mengunduhnya dari `repository` YOLO atau melatih model kustom.

### 8. Jalankan `Backend` FastAPI

Setelah semua prasyarat di atas terpenuhi dan `virtual environment` Anda aktif, jalankan `backend` FastAPI:

```bash
# Pastikan Anda berada di root folder proyek (drone-vision-be)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*   `--reload`: Memulai ulang `server` secara otomatis saat perubahan kode terdeteksi (berguna untuk pengembangan).
*   `--host 0.0.0.0`: Membuat `server` dapat diakses dari jaringan lokal Anda.
*   `--port 8000`: Menjalankan `server` di `port` 8000 (pastikan ini cocok dengan `BACKEND_WS_URL` di `frontend` Anda).

Anda akan melihat `log` saat `server` memulai, memuat model YOLO, dan `dataset` wajah.

---

## 💻 Gambaran Umum Project dari Sisi `Backend`

`Backend` ini dibangun menggunakan **FastAPI** (Python) untuk menyediakan `API` dan `WebSocket` `server` yang cepat dan `asynchronous`. Perannya adalah sebagai "otak" dari sistem `Drone Vision`, di mana semua pemrosesan `computer vision` yang intensif dilakukan.

### Arsitektur `Backend`

1.  **Server FastAPI (main.py):**
    *   Menerima `stream` video dari drone (atau `RTMP stream` simulasi dari OBS) melalui koneksi `WebSocket`.
    *   Melakukan semua pemrosesan `computer vision` pada `frame` video tersebut.
    *   Mengirimkan `frame` yang sudah di-`overlay` dengan hasil analisis, beserta `metadata` analitik (jumlah orang, kepadatan, wajah yang dikenali), kembali ke `frontend` melalui `WebSocket` terpisah.

2.  **Pemrosesan `Computer Vision`:**
    *   **Deteksi Objek (YOLO):** Menggunakan `model YOLO` (`ultralytics.YOLO`) untuk mendeteksi `objek orang` dalam setiap `frame`.
    *   **Pengenalan Wajah (FaceNet):** Menggunakan `library face_recognition` untuk mendeteksi wajah dan mengenali individu yang terdaftar di `dataset` wajah. `Embedding` wajah dimuat dari `cache` (`.pkl file`) untuk `startup` yang cepat.
    *   **Penghitungan Kepadatan Kerumunan:** Menganalisis `bounding box` orang dan informasi spasial (`grid`) untuk mengestimasi dan memberikan `status` kepadatan (Normal, Terdeteksi, Padat).
    *   **Visualisasi (`OpenCV`):** Hasil deteksi, pengenalan, dan kepadatan digambarkan (`overlay`) langsung pada `frame` video menggunakan `OpenCV`.

3.  **Optimalisasi Kinerja (`Asynchronous Processing`):**
    *   **`asyncio.to_thread()`:** Ini adalah kunci utama kinerja. Semua operasi `blocking` dari `OpenCV` (membaca `frame`, `resize`, `encode`, `drawing`) dan `library machine learning` (`YOLO.predict()`, `face_recognition.face_locations()`) dijalankan di `thread pool` terpisah menggunakan `await asyncio.to_thread()`. Ini memastikan `event loop` utama FastAPI tetap bebas dan responsif, mencegah `lag` parah pada `stream`.
    *   **Pemuatan Model Global:** Model YOLO dan `dataset` wajah dimuat **sekali** saat `server startup` (`@app.on_event("startup")`) untuk menghindari `loading` yang berulang.
    *   **Pengaturan Kualitas `Stream` Dinamis:** `Backend` menerima `parameter` dari `frontend` (Target FPS, Kualitas JPEG, Faktor `Resize Output`, Faktor `Resize` `CV Processing`) untuk memungkinkan kontrol `real-time` atas `trade-off` antara kualitas gambar dan performa.
    *   **`Client-Specific State`:** Status pemrosesan `CV` (`frame_count` untuk deteksi wajah berkala, `face_locations` sebelumnya) disimpan secara terpisah untuk setiap `client` `WebSocket` yang terhubung.

### Struktur Modul Utama (`app/main.py`)

*   **`@app.on_event("startup")`:** Memuat `yolov8n.pt` dan `dataset` wajah (`face_encodings_cache.pkl`) ke memori.
*   **`@app.on_event("shutdown")`:** Membersihkan `resource` saat `server` dimatikan.
*   **`@app.websocket("/ws/video_stream")`:** Endpoint `WebSocket` utama yang menangani koneksi `client` dan perintah (`start_stream`, `stop_stream`). Ini membuat dan mengelola `asyncio.Task` untuk setiap `stream` aktif.
*   **`async def capture_and_stream_frames(...)`:** Fungsi `async` yang dijalankan sebagai `task` untuk setiap `client`. Ia membuka `stream` video, membaca `frame` dalam `loop`, dan memanggil fungsi `processing` `blocking`.
*   **`def process_cv_frame_blocking(...)`:** Fungsi `blocking` yang berisi semua `logic computer vision` (YOLO, `face_recognition`, `grid density`, `annotasi OpenCV`). Ini dipanggil oleh `capture_and_stream_frames` melalui `asyncio.to_thread()`.
*   **`def encode_frame_to_jpg_blocking(...)`:** Fungsi `utility blocking` untuk mengubah ukuran dan meng-`encode` `frame` ke `JPEG` `base64`. Digunakan untuk `output final` ke `frontend`.
*   **`load_known_faces_blocking(...)` & `get_dataset_last_modified(...)`:** Fungsi `blocking` untuk mengelola `database` wajah dan `cache`nya.

---