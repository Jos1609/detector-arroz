"""
Lectura de imagenes con soporte opcional para HEIC/HEIF.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - depende del entorno local
    register_heif_opener = None


HEIC_EXTS = {".heic", ".heif"}


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_image(path: str | Path) -> np.ndarray | None:
    image_path = Path(path)
    suffix = image_path.suffix.lower()

    if suffix in HEIC_EXTS:
        if register_heif_opener is None:
            raise RuntimeError(
                "Para abrir imagenes HEIC/HEIF instala 'pillow-heif' y vuelve a intentar."
            )
        register_heif_opener()
        with Image.open(image_path) as image:
            return _pil_to_bgr(image)

    image = cv2.imread(str(image_path))
    if image is not None:
        return image

    try:
        with Image.open(image_path) as pil_image:
            return _pil_to_bgr(pil_image)
    except Exception:
        return None
