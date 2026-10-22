# Face Attendance (Python)

A browser UI backed by FastAPI. The browser captures a camera frame; Python detects one face and computes its FaceNet512 embedding with DeepFace. MySQL stores student and attendance records. Milvus stores and searches embeddings using cosine similarity.

## Run with Docker Compose

1. Copy `.env.example` to `.env` and set a strong `MYSQL_ROOT_PASSWORD`.
2. Start everything from this directory: `docker compose up --build`.
3. Open `http://localhost:8000/register.html` to register a student and capture one face.
4. Open `http://localhost:8000/attendance.html` to scan faces and mark attendance.
5. Open Attu at `http://localhost:8001` to inspect Milvus. It is preconfigured to connect to `milvus:19530`; browse the `student_faces_facenet512` collection to see saved student IDs and embeddings.
6. Health check: `http://localhost:8000/api/health`; API docs: `http://localhost:8000/docs`.

The first face request initializes/downloads the DeepFace model and can take several minutes. Model weights are not included in the archive. The app needs outbound network access for the first download; retain the model cache for offline restarts.

## Run the Python service locally

Start MySQL and Milvus first. From `server/`, run:

```sh
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Local defaults connect to MySQL and Milvus on `localhost`; set `DB_PASSWORD` and any overrides in `server/.env`.

## Request flow

- `POST /students`: save student details in MySQL.
- `POST /students/{id}/register-face` (multipart field `image`): Python validates the image, requires exactly one face, generates a 512-dimensional FaceNet512 embedding, and upserts it in Milvus.
- `POST /attendance/mark` (multipart field `image`): generate the same embedding, search Milvus, look up the matched student, and record at most one attendance row per student per local calendar day.
- `GET /students` and `GET /students/{id}`: list or fetch student records.

The frontend no longer loads face-api.js or generates descriptors. Detection and embedding generation happen in Python.

## Matching and migration

Milvus collection `student_faces_facenet512` has 512 dimensions and uses `COSINE`. Higher cosine scores indicate closer matches. `RECOGNITION_THRESHOLD` defaults to `0.60` as a demonstration value only; calibrate it using representative validation images before relying on attendance results.

The original project stored 128-dimensional face-api.js descriptors. They cannot be compared with FaceNet512 embeddings. This implementation uses a separate Milvus collection; re-enroll students through the Python service instead of copying the old vectors.

`Base.metadata.create_all` creates missing tables for local use. Add schema migrations before evolving a deployed schema. The compose file exposes infrastructure ports; restrict them and add authentication/TLS before exposing the stack beyond a trusted local network. Treat face embeddings as sensitive biometric data and define consent, access, retention, and deletion controls.
