import cv2
import numpy as np
from deepface import DeepFace
from fastapi import HTTPException, UploadFile

from .config import settings


async def embedding_from_upload(upload: UploadFile) -> list[float]:
    if upload.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    raw = await upload.read(settings.max_image_bytes + 1)
    if not raw or len(raw) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image is empty or exceeds the 8 MB limit")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    try:
        faces = DeepFace.represent(
            img_path=image,
            model_name=settings.face_model,
            detector_backend=settings.face_detector,
            enforce_detection=True,
            align=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="No face detected; center one face in the image") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Face model failed to process the image") from exc
    if len(faces) != 1:
        raise HTTPException(status_code=400, detail="Image must contain exactly one face")
    vector = [float(x) for x in faces[0]["embedding"]]
    if len(vector) != settings.face_dimension or not np.isfinite(vector).all():
        raise HTTPException(status_code=500, detail="Face model returned an invalid embedding")
    return vector
