from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path

from flask import current_app
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import detect_web
import detector_ia

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def ensure_storage_dirs():
    current_app.config["APP_STORAGE_DIR"].mkdir(parents=True, exist_ok=True)
    current_app.config["UPLOADS_DIR"].mkdir(parents=True, exist_ok=True)
    current_app.config["RESULTS_DIR"].mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_model_bundle():
    return detector_ia.cargar_modelo()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def process_sample_upload(file: FileStorage, lot_code: str, sample_number: int, confidence: float | None = None) -> dict:
    ensure_storage_dirs()
    confidence = detector_ia.CONF_MINIMA if confidence is None else confidence

    token = uuid.uuid4().hex[:12]
    safe_name = secure_filename(file.filename or f"muestra_{sample_number}.jpg")
    upload_name = f"{token}_{safe_name}"
    result_stem = f"{token}_{lot_code}_muestra_{sample_number}"

    upload_path = current_app.config["UPLOADS_DIR"] / upload_name
    result_image_path = current_app.config["RESULTS_DIR"] / f"{result_stem}.jpg"
    result_json_path = current_app.config["RESULTS_DIR"] / f"{result_stem}.json"

    file.save(upload_path)
    model, model_mode = get_model_bundle()
    result = detect_web.procesar_archivo(
        input_path=upload_path,
        output_json=result_json_path,
        output_image=result_image_path,
        sample_code=result_stem,
        source="rice_app",
        confidence=confidence,
        model=model,
        model_mode=model_mode,
    )

    return {
        "result": result,
        "upload_name": upload_name,
        "result_image_name": result_image_path.name,
        "result_json_name": result_json_path.name,
    }
