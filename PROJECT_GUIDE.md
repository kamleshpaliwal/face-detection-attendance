# Face Attendance Enterprise (Python): Build Guide and Production Roadmap

This document has two parts:

1. **Part A:** what was built from step 0 to today, with the *what*, *why* and *how* of each step.
2. **Part B:** a prioritized plan to make the project production ready.

> Note on history: git holds only two commits (`Initial import of face attendance app` and `chore: activity sync`). The steps below are reconstructed from the code and the README. They describe the logical build order, not a commit-by-commit history.

---

## Part A: What was implemented

### Step 0: Problem and design decisions

**What:** A system where a student registers once with a face photo. Later, a camera scan marks attendance automatically.

**Why these choices:**
| Decision | Reason |
|---|---|
| Face recognition in **Python/DeepFace**, not in the browser | The original version used face-api.js, which produced 128-dimension descriptors in the browser. Server-side FaceNet512 is more accurate and keeps the model under your control. The client can't tamper with the embeddings. |
| **MySQL** for students and attendance | This data is relational, needs uniqueness constraints and is easy to report on. |
| **Milvus** for embeddings | A vector database does fast nearest-neighbour search, so matching doesn't scan every row. |
| **Docker Compose** | One command brings up the DB, vector DB, API and admin UI. |

**How:** The architecture is `Browser -> FastAPI -> (DeepFace, MySQL, Milvus)`.

---

### Step 1: Configuration (`server/app/config.py`, `.env.example` files)

**What:** A `Settings` class built on `pydantic-settings`. It holds DB credentials, the Milvus address and collection name, the face model, the detector, the embedding dimension (512), the recognition threshold (0.60), the timezone, the maximum image size (8 MB), the CORS origins and the client directory.

**Why:** No secrets or environment-specific values go in code. The same image runs locally and in Docker with different env vars.

**How:** Settings load from environment variables or `server/.env`. The `database_url` property URL-encodes the password with `quote_plus`, so special characters don't break the connection string. `.gitignore` excludes `.env` files and keeps only `.env.example`.

---

### Step 2: Relational layer (`database.py`, `models.py`, `schemas.py`)

**What:**
- `database.py` creates the SQLAlchemy engine and the `get_db` session dependency. The engine uses `pool_pre_ping` and `pool_recycle=1800`.
- `models.py` defines `Student` and `Attendance`.
  - `Student` fields: `studentCode` (unique), first and last name, dob, className, section, createdAt.
  - `Attendance` fields: studentId (FK), date, time, confidence.
  - `Attendance` has the unique constraint `uq_attendance_student_date` on (studentId, date).
- `schemas.py` has the Pydantic request and response models: `StudentCreate` (with length validation), `StudentOut` and `AttendanceOut`.

**Why:**
- The unique constraint enforces "one attendance row per student per day" in the database. That holds even if two scans arrive at the same moment.
- `pool_pre_ping` and `pool_recycle` avoid "MySQL server has gone away" errors on stale connections.
- The column names (`studentCode`, `firstName` and so on) match the earlier version of the app, so the schema stays compatible.

**How:** `Base.metadata.create_all` runs at startup and creates any missing tables.

---

### Step 3: Face embedding pipeline (`server/app/faces.py`)

**What:** `embedding_from_upload` turns an uploaded image into a 512-float vector.

**Why:** Registration and attendance must produce comparable vectors, so both use this one function. It also keeps bad input out of the model.

**How, in order:**
1. Check the content type (JPEG, PNG or WebP only) and return 415 otherwise.
2. Read at most `max_image_bytes + 1` bytes. Empty or oversize files get 413, and the cap protects memory.
3. Decode with OpenCV. Undecodable data gets 400.
4. Call `DeepFace.represent` with `Facenet512`, the `opencv` detector, `enforce_detection=True` and `align=True`.
5. No face returns 400. A model failure returns 503.
6. Require **exactly one face**, so a group photo can't enroll or mark the wrong person.
7. Check that the vector has 512 dimensions and that every value is finite.

---

### Step 4: Vector layer (`server/app/vectors.py`)

**What:** A `FaceVectors` class that wraps `MilvusClient`.

**Why:** Milvus keeps similarity search out of MySQL and scales better as the student count grows.

**How:**
- `initialize()` creates the collection `student_faces_facenet512`. Its primary key is `student_id` (INT64) and it has a 512-dimension `embedding` field. The index is **HNSW** (M=16, efConstruction=200) with the **COSINE** metric. It then loads the collection into memory.
- `put()` upserts, so re-registering a student replaces their embedding.
- `nearest()` runs a top-1 search (ef=64) and returns `{student_id, score}`.
- The collection name includes the model name. The old 128-dimension vectors can't be compared with FaceNet512, so students must re-enroll.

---

### Step 5: API (`server/app/main.py`)

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness check that also reports the model and dimension |
| `POST /students` | Create a student. A duplicate `studentCode` returns 409. |
| `GET /students`, `GET /students/{id}` | List students or fetch one. A missing student returns 404. |
| `POST /students/{id}/register-face` | Embed the uploaded face and upsert it into Milvus |
| `POST /attendance/mark` | Embed, search, apply the threshold, then record attendance |

**Why the design is this way:**
- **Lifespan startup:** it creates tables and retries Milvus up to 12 times, 5 seconds apart. Milvus starts slowly, and this stops the app crashing if it comes up first.
- **Threshold check:** a nearest neighbour always exists, even for a stranger. The score must reach `recognition_threshold` (cosine, higher is better) before it counts as a match.
- **Idempotent attendance:** the code checks for an existing row, and also catches `IntegrityError` for the race case. In both cases it returns "already marked".
- **Timezone-aware "today":** the date and time come from `APP_TIMEZONE`, so "per day" matches the school's calendar day.
- **CORS** is limited to configured origins and to GET and POST.
- **Static hosting:** the app mounts `/client`, so one container serves both the UI and the API.

---

### Step 6: Frontend (`client/register.html`, `client/attendance.html`)

**What:**
- `register.html` has a student form. Submitting it calls `POST /students`, then a camera preview captures a frame and calls `register-face`.
- `attendance.html` opens the camera and sends a JPEG frame to `/attendance/mark` every 2 seconds. It shows the result.

**Why:** Plain HTML and JavaScript need no build step. Because the browser only sends images, all recognition logic stays on the server.

**How:** `getUserMedia` gives the video stream. A canvas `drawImage` and `toBlob` (JPEG, quality 0.85 to 0.9) produce the image. It goes up as multipart `FormData` with the field name `image`. `API_URL` is empty, so requests are same-origin.

---

### Step 7: Containerization (`server/Dockerfile`, `docker-compose.yml`, `milvus/`)

**What:**
- The Dockerfile uses `python:3.11-slim` and installs the OpenCV system libraries (`libgl1`, `libglib2.0-0`, `libgomp1`). It installs dependencies before copying the code, so layer caching works.
- Compose runs four services:
  - `mysql` has a healthcheck.
  - `milvus` is standalone with **embedded etcd and local storage**. Embedded etcd uses `milvus/embedEtcd.yaml`. This removes the separate etcd and MinIO containers.
  - `attu` is the Milvus web UI on port 8001.
  - `app` waits for both `mysql` and `milvus` to be healthy.
- A `deepface_cache` volume keeps the model weights across restarts, so they aren't downloaded again.
- `${MYSQL_ROOT_PASSWORD:?...}` makes Compose fail fast if the password isn't set.

**Why:** A simpler stack (3 infrastructure containers instead of 5) is easier to run locally. The caching and healthchecks make startup reliable.

---

### Step 8: Latest, uncommitted changes (current working tree)

- `docker-compose.yml` was changed to the embedded-etcd Milvus setup described above.
- `milvus/embedEtcd.yaml` was added (**untracked**).
- `server/.dockerignore` was added (**untracked**). It keeps `.env`, virtualenvs and caches out of the image.
- Action: commit these. Without them, a fresh clone can't start Milvus.

---

## Part B: Production-readiness plan

Priorities: **P0** = blocker before real users, **P1** = needed soon after, **P2** = improvement.

### Phase 1: Security and privacy (P0)

| # | Task | Why | How |
|---|---|---|---|
| 1.1 | **Authentication and roles** (admin, teacher, kiosk) | Today anyone can register students or mark attendance. | JWT or OAuth2 through FastAPI dependencies. Admins manage students. A kiosk gets a scoped API key that can only call `/attendance/mark`. |
| 1.2 | **HTTPS** | Camera access (`getUserMedia`) requires HTTPS outside localhost, and biometrics must be encrypted in transit. | Put Caddy, Nginx or Traefik in front. Don't publish port 8000 directly. |
| 1.3 | **Stop exposing infrastructure ports** | 3306, 19530, 9091 and Attu are open to the host. | Remove the `ports:` entries for MySQL and Milvus. Put Attu behind auth or an SSH tunnel. |
| 1.4 | **Dedicated DB user** | The app connects as `root`. | Create a least-privilege user limited to its own database. |
| 1.5 | **Secrets management** | `.env` files are fragile. | Docker secrets, or Vault or a cloud secret manager. |
| 1.6 | **Liveness / anti-spoofing** | A printed photo or phone screen can mark someone present. | DeepFace `anti_spoofing=True`, or blink and head-turn challenges. Test with real spoof attempts. |
| 1.7 | **Biometric compliance** | Embeddings are sensitive personal data. | Consent capture, an access audit log, a retention policy, and a delete endpoint that removes the student from MySQL and Milvus together. Check the local law (for example India's DPDP Act, GDPR). |
| 1.8 | **Rate limiting** | Each scan is expensive, so the endpoint is easy to flood. | `slowapi` or limits at the proxy. |

### Phase 2: Correctness and accuracy (P0 to P1)

| # | Task | Why | How |
|---|---|---|---|
| 2.1 | **Calibrate the threshold** | 0.60 is only a demo value. | Collect a labeled validation set. Plot the false accept and false reject rates. Pick the threshold that matches the school's risk tolerance. |
| 2.2 | **Stronger detector** | The `opencv` (Haar) detector is weak with angles and lighting. | Try `retinaface`, `mtcnn` or `yolov8`. Benchmark speed against accuracy. |
| 2.3 | **Multiple embeddings per student** | One photo is fragile. | Store 3 to 5 enrollment images. Change the Milvus primary key to an auto ID with a `student_id` field. Match on the best score or the average of the top scores. |
| 2.4 | **Ambiguity margin** | Two close students can cause a wrong match. | Take the top 2 results. Reject the scan if the gap between them is too small. |
| 2.5 | **Consistent state** | Registration writes to MySQL and Milvus separately, so a failure can leave them out of sync. | A reconciliation job that finds students with no vector, or vectors with no student. Delete in both stores in order. |
| 2.6 | **Bug and cleanup fixes** | Deprecated `datetime.utcnow`; the app connects to Milvus at import time; `create_all` is used instead of migrations. | Use timezone-aware `datetime.now(UTC)`. Create the Milvus client in `lifespan`. Adopt Alembic. |
| 2.7 | **Return the right HTTP status** | `/attendance/mark` returns 200 with `success: false` for "not recognized". | Use consistent status codes (404 or 422) and an error schema. |

### Phase 3: Features (P1)

- Attendance read APIs: by date, class or student; daily and monthly summaries; CSV and Excel export.
- Student update, delete and soft-delete. Re-enroll a face. Bulk import from CSV.
- Manual attendance override by a teacher, with a reason and an audit trail.
- Absent and late logic: school hours, cut-off times, per-class schedules.
- Admin dashboard (React, or the existing HTML grown into one). It needs login and student and attendance tables.
- Notifications (SMS, email or WhatsApp) to parents for absences.
- Pagination and search on `/students`.

### Phase 4: Performance and scalability (P1)

- Model warm-up at startup. The first request currently takes minutes.
- Run DeepFace in a thread pool or process pool. Right now `DeepFace.represent` blocks the async event loop.
- Several Uvicorn or Gunicorn workers. Move to GPU inference if volume demands it.
- Client-side change: add a client-side face-presence check, throttle the 2-second polling and skip frames with no face.
- Resize images client-side before upload.
- Real Milvus deployment (cluster, or standalone with separate etcd and MinIO) and MySQL backups or a managed DB. Embedded etcd and local storage suit development only.

### Phase 5: Reliability and operations (P1)

- **Backups:** scheduled MySQL dumps and Milvus snapshots, with a tested restore.
- **Logging:** structured JSON logs and request IDs. Never log images or embeddings.
- **Monitoring:** Prometheus metrics (latency, match rate, error rate), Grafana dashboards and alerts. Extend `/api/health` to check MySQL and Milvus connectivity (readiness vs liveness).
- **Container hardening:** run as a non-root user, pin base image versions, and scan images with Trivy.
- **Resource limits and restart policies** in Compose, or move to Kubernetes.

### Phase 6: Quality and delivery (P1 to P2)

- **Tests:** pytest unit tests (schemas, threshold logic, duplicate handling). API tests with a mocked DeepFace and Milvus. Integration tests through Compose. A regression set of face images.
- **CI/CD (GitHub Actions):** lint (ruff), type check (mypy), tests, image build and scan, then deploy.
- **Migrations:** Alembic, versioned.
- **Config and docs:** separate dev, staging and prod configs. Add OpenAPI examples, a runbook and a README section for deployment.
- **Pin dependencies:** a lock file. `deepface>=0.0.93` is unbounded upward.

### Suggested order

1. **Week 1:** commit the pending files. Complete 1.1 to 1.4. Complete 2.6. Add the basic tests.
2. **Week 2:** 1.2, 1.6, 1.8 and 2.1 to 2.4, with the accuracy validation set.
3. **Week 3:** Phase 3 reports and the admin features. Phase 4 performance work.
4. **Week 4:** Phase 5 operations, CI/CD and a staging pilot with one class before wider rollout.

### Definition of "production ready"

- [ ] All endpoints authenticated, and traffic over HTTPS only
- [ ] No DB or vector ports exposed publicly
- [ ] The threshold calibrated on real data, with measured false accept and false reject rates
- [ ] Anti-spoofing enabled
- [ ] A consent record and a deletion path for biometric data
- [ ] Backups run and the restore has been tested
- [ ] Monitoring and alerts in place
- [ ] Automated tests and a CI pipeline passing
- [ ] A pilot completed with real users
