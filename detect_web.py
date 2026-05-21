"""
Deteccion web de granos de arroz.

Uso:
    python detect_web.py --input in.jpg --output-json out.json --output-image out.jpg
"""

import argparse
import base64
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import detector_ia
import image_io

if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent / "web" / "storage")

from ultralytics import YOLO

MODELO_PATH = "modelo_arroz.pt"
MIN_AREA_GRANO = 700
MAX_AREA_FRAC = 0.60
SUPPLEMENT_MIN_CONF = 0.55
SUPPLEMENT_IOU = 0.35
SUPPLEMENT_MAX_EXTRA = 6

COLORES_BGR = {
    "sano": (50, 205, 50),
    "panza_blanca": (0, 60, 255),
    "quebrado": (200, 200, 200),
}

ETIQUETAS = {
    "sano": "SANO",
    "panza_blanca": "PANZA BLANCA",
    "quebrado": "QUEBRADO",
}


def detectar_granos(frame: np.ndarray):
    alto, ancho = frame.shape[:2]
    area_total = alto * ancho
    min_area = max(MIN_AREA_GRANO, int(area_total * 0.00008))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def extraer_granos(mask: np.ndarray):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        granos_local = []
        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > ancho * 0.85 or h > alto * 0.85:
                continue
            bbox_area = w * h
            if bbox_area < min_area:
                continue
            if bbox_area > area_total * MAX_AREA_FRAC:
                continue

            contour_area = cv2.contourArea(cnt)
            if contour_area < max(450, min_area * 0.55):
                continue

            fill_ratio = contour_area / max(1.0, float(bbox_area))
            aspect_ratio = max(w, h) / max(1.0, min(w, h))
            if fill_ratio < 0.28:
                continue
            if aspect_ratio > 2.7 and contour_area < 2500:
                continue
            if aspect_ratio > 3.3:
                continue

            roi_hsv = hsv[y:y + h, x:x + w]
            mean_sat = float(np.mean(roi_hsv[:, :, 1]))
            mean_val = float(np.mean(roi_hsv[:, :, 2]))
            if mean_sat < 8 and mean_val > 170:
                continue
            if mean_sat < 5 and mean_val > 150 and contour_area < 1200:
                continue

            granos_local.append((x, y, w, h))
        return granos_local

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    normalizada = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(normalizada, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_adapt = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 6
    )
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_or(thresh_adapt, thresh_otsu)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh_legacy = cv2.adaptiveThreshold(
        cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10,
    )
    thresh_legacy = cv2.morphologyEx(thresh_legacy, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh_legacy = cv2.morphologyEx(thresh_legacy, cv2.MORPH_OPEN, kernel, iterations=1)

    borde = 10
    for mask in (thresh, thresh_legacy):
        mask[:borde, :] = 0
        mask[-borde:, :] = 0
        mask[:, :borde] = 0
        mask[:, -borde:] = 0

    granos_nuevo = extraer_granos(thresh)
    granos_legacy = extraer_granos(thresh_legacy)
    return granos_nuevo if len(granos_nuevo) >= len(granos_legacy) else granos_legacy


def ordenar_puntos(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def recorte_rotado_desde_caja(frame: np.ndarray, x: int, y: int, w: int, h: int):
    margen = 10
    alto, ancho = frame.shape[:2]
    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(ancho, x + w + margen)
    y2 = min(alto, y + h + margen)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return roi

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return roi

    centro_roi = np.array([roi.shape[1] / 2.0, roi.shape[0] / 2.0], dtype=np.float32)

    def puntaje_contorno(cnt):
        area = cv2.contourArea(cnt)
        if area <= 0:
            return -1e9
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        centro = np.array([rx + rw / 2.0, ry + rh / 2.0], dtype=np.float32)
        dist = np.linalg.norm(centro - centro_roi)
        return area - dist * 4.0

    cnt = max(contornos, key=puntaje_contorno)
    if len(cnt) < 5:
        return roi

    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    if rw < 2 or rh < 2:
        return roi

    box = cv2.boxPoints(rect)
    box = ordenar_puntos(box)
    out_w = max(1, int(round(max(rw, rh))))
    out_h = max(1, int(round(min(rw, rh))))

    destino = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matriz = cv2.getPerspectiveTransform(box.astype(np.float32), destino)
    warp = cv2.warpPerspective(roi, matriz, (out_w, out_h), borderValue=(255, 255, 255))

    if warp.shape[0] > warp.shape[1]:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

    return warp


def clasificar_recorte(model, frame, x, y, w, h):
    recorte = recorte_rotado_desde_caja(frame, x, y, w, h)
    if recorte.size == 0:
        return None, 0.0
    classifier = model.get("classifier") if isinstance(model, dict) else model
    if classifier is None:
        return None, 0.0
    results = classifier(recorte, verbose=False)
    probs = results[0].probs
    clase_id = int(probs.top1)
    confianza = float(probs.top1conf)
    clase_str = model.names[clase_id]
    return clase_str, confianza


def calcular_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suplementar_detecciones(
    frame: np.ndarray,
    classifier,
    detecciones_detector: list[dict],
    conf_min: float,
    objetivo_extra: int,
):
    if classifier is None:
        return []
    if objetivo_extra <= 0 or objetivo_extra > 3:
        return []

    cajas_detector = [tuple(int(v) for v in det["bbox"]) for det in detecciones_detector]
    adicionales = []
    umbral_conf = max(conf_min, SUPPLEMENT_MIN_CONF)

    for (x, y, w, h) in detectar_granos(frame):
        bbox = (x, y, x + w, y + h)
        if any(calcular_iou_xyxy(bbox, box_det) >= SUPPLEMENT_IOU for box_det in cajas_detector):
            continue

        clase, conf = clasificar_recorte(classifier, frame, x, y, w, h)
        if clase is None or conf < umbral_conf:
            continue

        adicionales.append({"clase": clase, "conf": float(conf), "bbox": bbox})

    adicionales.sort(key=lambda det: det["conf"], reverse=True)
    limite = min(SUPPLEMENT_MAX_EXTRA, max(0, int(objetivo_extra)))
    return adicionales[:limite]


def elegir_mejor_resultado(resultado_detect, resultado_legacy):
    frame_det, conteos_det, promedio_det = resultado_detect
    frame_leg, conteos_leg, promedio_leg = resultado_legacy

    total_det = int(sum(conteos_det.values()))
    total_leg = int(sum(conteos_leg.values()))

    # Preferimos el metodo legado cuando recupera algunos granos extra
    # sin dispararse a un conteo absurdo como pasaba en los peores casos.
    limite_leg = max(total_det + 6, int(total_det * 1.6))
    legado_usable = (
        total_leg > total_det
        and total_leg <= limite_leg
        and promedio_leg >= max(0.45, promedio_det - 0.12)
    )

    if legado_usable:
        return frame_leg, conteos_leg, promedio_leg, "legacy_classify"
    return frame_det, conteos_det, promedio_det, "detect_hybrid"


def procesar_frame(
    frame: np.ndarray,
    sample_id: str = "",
    sample_code: str = "",
    producer_name: str = "",
    source: str = "web",
    confidence: float = detector_ia.CONF_MINIMA,
    model=None,
    model_mode: str | None = None,
):
    model, model_mode = (model, model_mode) if model is not None and model_mode is not None else detector_ia.cargar_modelo()
    selected_mode = model_mode

    if frame is None:
        raise RuntimeError("No se pudo leer la imagen de entrada.")

    alto, ancho = frame.shape[:2]
    if ancho > 1920 or alto > 1080:
        escala = min(1920 / ancho, 1080 / alto)
        frame = cv2.resize(frame, (0, 0), fx=escala, fy=escala)

    if str(model_mode).startswith("detect") or str(model_mode) == "hybrid":
        detecciones_det = detector_ia.predecir_detecciones(model, frame.copy(), confidence)
        frame_annotated, conteos, confs_validas = detector_ia.resumir_detecciones(frame.copy(), detecciones_det)
        promedio = float(sum(confs_validas) / len(confs_validas)) if confs_validas else 0.0
        detecciones_payload = [
            {
                "clase": det["clase"],
                "conf": float(det["conf"]),
                "x1": int(det["bbox"][0]),
                "y1": int(det["bbox"][1]),
                "x2": int(det["bbox"][2]),
                "y2": int(det["bbox"][3]),
            }
            for det in detecciones_det
        ]
    else:
        frame_annotated, conteos, promedio = analizar_imagen(model, frame, confidence)
        detecciones_payload = []

    payload = {
        "sample_id": sample_id,
        "sample_code": sample_code,
        "producer_name": producer_name,
        "captured_at": datetime.now().isoformat(),
        "source": source,
        "model_mode": model_mode,
        "selected_mode": selected_mode,
        "counts": conteos,
        "total_detected": int(sum(conteos.values())),
        "average_confidence": promedio,
        "detections": detecciones_payload,
        "result_saved": False,
    }
    return payload, frame_annotated


def analizar_imagen(model, image_input, conf_min: float):
    if isinstance(image_input, np.ndarray):
        frame = image_input.copy()
    else:
        frame = image_io.read_image(image_input)
    if frame is None:
        raise RuntimeError("No se pudo leer la imagen de entrada.")

    alto, ancho = frame.shape[:2]
    if ancho > 1920 or alto > 1080:
        escala = min(1920 / ancho, 1080 / alto)
        frame = cv2.resize(frame, (0, 0), fx=escala, fy=escala)

    conteos = {"sano": 0, "panza_blanca": 0, "quebrado": 0}
    confs_validas = []
    granos = detectar_granos(frame)

    for (x, y, w, h) in granos:
        clase, conf = clasificar_recorte(model, frame, x, y, w, h)
        if clase is None or conf < conf_min:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 2)
            continue

        color = COLORES_BGR.get(clase, (200, 200, 200))
        label = f"{ETIQUETAS.get(clase, clase)} {conf:.0%}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y - 5, th + 5)
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 2), color, -1)
        cv2.putText(frame, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        conteos[clase] += 1
        confs_validas.append(conf)

    promedio = float(sum(confs_validas) / len(confs_validas)) if confs_validas else 0.0
    return frame, conteos, promedio


def frame_to_data_url(frame: np.ndarray, quality: int = 82) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen procesada.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def procesar_archivo(
    input_path: Path,
    output_json: Path,
    output_image: Path,
    sample_id: str = "",
    sample_code: str = "",
    producer_name: str = "",
    source: str = "web",
    confidence: float = detector_ia.CONF_MINIMA,
    model=None,
    model_mode: str | None = None,
):
    frame = image_io.read_image(input_path)
    result_payload, frame = procesar_frame(
        frame=frame,
        sample_id=sample_id,
        sample_code=sample_code,
        producer_name=producer_name,
        source=source,
        confidence=confidence,
        model=model,
        model_mode=model_mode,
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), frame)
    result_payload["preview_image"] = str(output_image.resolve())
    result_payload["result_saved"] = True

    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(result_payload, fh, ensure_ascii=False, indent=2)

    return result_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--sample-code", default="")
    parser.add_argument("--producer-name", default="")
    parser.add_argument("--source", default="web")
    parser.add_argument("--confidence", type=float, default=detector_ia.CONF_MINIMA)
    args = parser.parse_args()

    model, model_mode = detector_ia.cargar_modelo()
    procesar_archivo(
        input_path=Path(args.input),
        output_json=Path(args.output_json),
        output_image=Path(args.output_image),
        sample_id=args.sample_id,
        sample_code=args.sample_code,
        producer_name=args.producer_name,
        source=args.source,
        confidence=args.confidence,
        model=model,
        model_mode=model_mode,
    )


if __name__ == "__main__":
    main()
