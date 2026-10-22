from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_host: str = "localhost"
    db_port: int = 3306
    db_username: str = "root"
    db_password: str = ""
    db_database: str = "student_attendance_face_db"
    milvus_address: str = "localhost:19530"
    milvus_collection: str = "student_faces_facenet512"
    face_model: str = "Facenet512"
    face_detector: str = "opencv"
    face_dimension: int = 512
    recognition_threshold: float = 0.60
    app_timezone: str = "Asia/Kolkata"
    max_image_bytes: int = 8 * 1024 * 1024
    cors_origins: str = "http://localhost:8000"
    client_dir: str = "../client"

    @property
    def database_url(self) -> str:
        from urllib.parse import quote_plus

        password = quote_plus(self.db_password)
        return (
            f"mysql+pymysql://{self.db_username}:{password}@{self.db_host}:"
            f"{self.db_port}/{self.db_database}?charset=utf8mb4"
        )


settings = Settings()
