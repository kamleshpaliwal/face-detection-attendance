from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict, Field


class StudentCreate(BaseModel):
    studentCode: str = Field(min_length=3, max_length=20)
    firstName: str = Field(min_length=1, max_length=100)
    lastName: str = Field(min_length=1, max_length=100)
    dob: date | None = None
    className: str | None = Field(default=None, max_length=100)
    section: str | None = Field(default=None, max_length=50)


class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    id: int
    studentCode: str = Field(validation_alias="student_code")
    firstName: str = Field(validation_alias="first_name")
    lastName: str = Field(validation_alias="last_name")
    dob: date | None
    className: str | None = Field(validation_alias="class_name")
    section: str | None
    createdAt: datetime = Field(validation_alias="created_at")


class AttendanceOut(BaseModel):
    id: int
    date: date
    time: time
    confidence: float
