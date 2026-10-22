from contextlib import asynccontextmanager
from datetime import datetime
import time as time_module
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import settings
from .database import Base, engine, get_db
from .faces import embedding_from_upload
from .models import Attendance, Student
from .schemas import StudentCreate, StudentOut
from .vectors import face_vectors


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    for attempt in range(12):
        try:
            face_vectors.initialize()
            break
        except Exception as exc:
            if attempt == 11:
                raise
            time_module.sleep(5)
    yield
    face_vectors.client.close()


app = FastAPI(title="Face Attendance API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health():
    return {"success": True, "model": settings.face_model, "dimension": settings.face_dimension}


@app.post("/students", response_model=StudentOut, status_code=201)
def create_student(payload: StudentCreate, db: Session = Depends(get_db)):
    student = Student(
        student_code=payload.studentCode,
        first_name=payload.firstName,
        last_name=payload.lastName,
        dob=payload.dob,
        class_name=payload.className,
        section=payload.section,
    )
    db.add(student)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student code already exists") from exc
    db.refresh(student)
    return student


@app.get("/students", response_model=list[StudentOut])
def list_students(db: Session = Depends(get_db)):
    return list(db.scalars(select(Student).order_by(Student.id)))


@app.get("/students/{student_id}", response_model=StudentOut)
def get_student(student_id: int, db: Session = Depends(get_db)):
    student = db.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@app.post("/students/{student_id}/register-face")
async def register_face(
    student_id: int, image: UploadFile = File(...), db: Session = Depends(get_db)
):
    student = db.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    vector = await embedding_from_upload(image)
    face_vectors.put(student_id, vector)
    return {"success": True, "message": "Face registered successfully"}


@app.post("/attendance/mark")
async def mark_attendance(image: UploadFile = File(...), db: Session = Depends(get_db)):
    vector = await embedding_from_upload(image)
    match = face_vectors.nearest(vector)
    if not match:
        return {"success": False, "message": "Student not recognized"}

    # COSINE scores are higher for more similar vectors. Calibrate this value
    # against representative validation images before production use.
    if match["score"] < settings.recognition_threshold:
        return {"success": False, "message": "Student not recognized (low similarity)", "score": match["score"]}

    student = db.get(Student, match["student_id"])
    if not student:
        return {"success": False, "message": "Student verification failed"}

    local_now = datetime.now(ZoneInfo(settings.app_timezone))
    today = local_now.date()
    existing = db.scalar(
        select(Attendance).where(
            Attendance.student_id == student.id,
            Attendance.date == today,
        )
    )
    if existing:
        return {"success": True, "message": "Attendance already marked", "data": _attendance_data(student, existing)}

    attendance = Attendance(
        student_id=student.id,
        date=today,
        time=local_now.time().replace(tzinfo=None, microsecond=0),
        confidence=match["score"],
    )
    db.add(attendance)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        attendance = db.scalar(
            select(Attendance).where(Attendance.student_id == student.id, Attendance.date == today)
        )
        return {"success": True, "message": "Attendance already marked", "data": _attendance_data(student, attendance)}
    db.refresh(attendance)
    return {"success": True, "message": "Attendance marked successfully", "data": _attendance_data(student, attendance)}


def _attendance_data(student: Student, attendance: Attendance) -> dict:
    return {
        "student": {
            "id": student.id,
            "studentCode": student.student_code,
            "firstName": student.first_name,
            "lastName": student.last_name,
            "className": student.class_name,
        },
        "attendance": {
            "id": attendance.id,
            "date": attendance.date.isoformat(),
            "time": attendance.time.isoformat(),
            "confidence": attendance.confidence,
        },
    }


app.mount("/", StaticFiles(directory=str(Path(settings.client_dir).resolve()), html=True), name="client")
