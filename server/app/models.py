from datetime import date, datetime, time

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Time, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_code: Mapped[str] = mapped_column("studentCode", String(20), unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column("firstName", String(100), nullable=False)
    last_name: Mapped[str] = mapped_column("lastName", String(100), nullable=False)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    class_name: Mapped[str | None] = mapped_column("className", String(100), nullable=True)
    section: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=datetime.utcnow)


class Attendance(Base):
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("studentId", "date", name="uq_attendance_student_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column("studentId", ForeignKey("students.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    time: Mapped[time] = mapped_column(Time, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    student: Mapped[Student] = relationship(lazy="joined")
