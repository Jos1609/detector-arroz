"""
Inferencia compartida para el detector de granos de arroz.

Prioriza un modelo YOLO de deteccion completa si existe:
    modelo_arroz_detect.pt

Si todavia no existe, puede caer al clasificador:
    modelo_arroz_classify.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

if "YOLO_CONFIG_DIR" not in os.environ:
    runtime_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir / "web" / "storage")

from ultralytics import YOLO

def _runtime_search_dirs() -> list[Path]:
    dirs: list[Path] = []

    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            dirs.append(Path(meipass))

    dirs.extend(
        [
            Path.cwd(),
            Path(__file__).resolve().parent,
        ]
    )

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for item in dirs:
        key = str(item.resolve()) if item.exists() else str(item)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(item)
    return unique_dirs


def _model_candidates(*names: str) -> list[Path]:
    candidates: list[Path] = []
    for base_dir in _runtime_search_dirs():
        for name in names:
            candidates.append(base_dir / name)
    return candidates


DETECT_MODEL_CANDIDATES = _model_candidates(
    "modelo_arroz_detect.pt",
)
CLASSIFY_MODEL_CANDIDATES = _model_candidates(
    "modelo_arroz_classify.pt",
)

# Con el detector puro, 0.20 recupera mejor los granos faltantes
# sin disparar el ruido que metia el flujo legacy.
CONF_MINIMA = 0.20
DETECT_IMGSZ = 1280
DETECT_IOU = 0.30
DETECT_MAX_DET = 800
TILE_OVERLAP_RATIO = 0.25
TILE_ASPECT_RATIO_TRIGGER = 1.15
CLASSIFY_MINIMA = 0.35
BOX_MARGIN = 10
DEDUP_CROSS_CLASS_IOU = 0.35 # Si se enciman 35%, se borra el duplicado
DEDUP_CONTAINMENT = 0.65     # Si un grano contiene al 65% de otro, se borra el duplicado

COLORES_BGR = {
    "sano": (50, 205, 50),
    "panza_blanca": (0, 60, 255),
    "quebrado": (0, 215, 255), # Amarillo Dorado para mejor visibilidad
}

COLORES_HEX = {
    "sano": "#27ae60",
    "panza_blanca": "#e74c3c",
    "quebrado": "#f1c40f",
}

ETIQUETAS = {
    "sano": "SANO",
    "panza_blanca": "PANZA BLANCA",
    "quebrado": "QUEBRADO",
}


def cargar_modelo():
    classifier = None
    for classify_path in CLASSIFY_MODEL_CANDIDATES:
        if classify_path.exists():
            classifier = YOLO(str(classify_path))
            break
    
    # Intentar cargar ambos para flujo híbrido
    detector = None
    for detect_path in DETECT_MODEL_CANDIDATES:
        if detect_path.exists():
            detector = YOLO(str(detect_path))
            break
            
    if detector and classifier:
        return {"detector": detector, "classifier": classifier}, "hybrid"
    elif detector:
        return {"detector": detector, "classifier": None}, "detect_only"
    elif classifier:
        return classifier, "classify"
    raise FileNotFoundError(
        "No se encontro ningun modelo. Busca: modelo_arroz_detect.pt o "
        "modelo_arroz_classify.pt"
    )


def ordenar_puntos(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def recorte_rotado_desde_caja(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    alto, ancho = frame.shape[:2]
    rx1 = max(0, x1 - BOX_MARGIN)
    ry1 = max(0, y1 - BOX_MARGIN)
    rx2 = min(ancho, x2 + BOX_MARGIN)
    ry2 = min(alto, y2 + BOX_MARGIN)
    roi = frame[ry1:ry2, rx1:rx2]
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
        bx, by, bw, bh = cv2.boundingRect(cnt)
        centro = np.array([bx + bw / 2.0, by + bh / 2.0], dtype=np.float32)
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


def clasificar_recorte(classifier, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    if classifier is None:
        return None, 0.0
    recorte = recorte_rotado_desde_caja(frame, x1, y1, x2, y2)
    if recorte.size == 0:
        return None, 0.0
    results = classifier(recorte, verbose=False)
    probs = results[0].probs
    clase_id = int(probs.top1)
    confianza = float(probs.top1conf)
    clase_str = classifier.names[clase_id]
    return clase_str, confianza


def dibujar_deteccion(frame: np.ndarray, clase: str, conf: float, x1: int, y1: int, x2: int, y2: int):
    color = COLORES_BGR.get(clase, (200, 200, 200))
    # Dibujar solo el rectángulo para que la imagen sea más clara
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def construir_tiles_cuadrados(frame: np.ndarray):
    alto, ancho = frame.shape[:2]
    lado = min(alto, ancho)
    if lado <= 0:
        return [(frame, 0, 0)]

    ratio = max(alto, ancho) / max(1, lado)
    if ratio < TILE_ASPECT_RATIO_TRIGGER:
        return [(frame, 0, 0)]

    paso = max(1, int(round(lado * (1.0 - TILE_OVERLAP_RATIO))))
    tiles = []

    if alto >= ancho:
        offsets = list(range(0, max(1, alto - lado + 1), paso))
        if offsets[-1] != alto - lado:
            offsets.append(alto - lado)
        for y0 in offsets:
            tiles.append((frame[y0:y0 + lado, 0:lado], 0, int(y0)))
    else:
        offsets = list(range(0, max(1, ancho - lado + 1), paso))
        if offsets[-1] != ancho - lado:
            offsets.append(ancho - lado)
        for x0 in offsets:
            tiles.append((frame[0:lado, x0:x0 + lado], int(x0), 0))

    return tiles


def iou_xyxy(box_a, box_b):
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


def interseccion_sobre_caja_menor(box_a, box_b):
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
    return inter / max(1, min(area_a, area_b))


def son_detecciones_duplicadas(actual: dict, previa: dict, iou_thr: float):
    bbox_actual = tuple(int(v) for v in actual["bbox"])
    bbox_previa = tuple(int(v) for v in previa["bbox"])
    iou = iou_xyxy(bbox_actual, bbox_previa)

    # 1. Check por traslape general (NMS estándar)
    thr = iou_thr if actual["clase"] == previa["clase"] else DEDUP_CROSS_CLASS_IOU
    if iou >= thr:
        return True

    # 2. Check por contención (un cuadro dentro de otro)
    # Esto mata los casos donde se detecta el grano y aparte una parte del grano
    return interseccion_sobre_caja_menor(bbox_actual, bbox_previa) >= DEDUP_CONTAINMENT


def deduplicar_detecciones(detecciones: list[dict], iou_thr: float):
    detecciones_ordenadas = sorted(detecciones, key=lambda det: float(det["conf"]), reverse=True)
    filtradas: list[dict] = []

    for det in detecciones_ordenadas:
        if any(son_detecciones_duplicadas(det, prev, iou_thr) for prev in filtradas):
            continue
        filtradas.append(det)

    return filtradas


def predecir_detecciones(models, frame: np.ndarray, conf_min: float):
    detector = models["detector"] if isinstance(models, dict) else models
    classifier = models.get("classifier") if isinstance(models, dict) else None

    tiles = construir_tiles_cuadrados(frame)
    inputs = [tile_frame for tile_frame, _, _ in tiles]
    results = detector.predict(
        inputs,
        conf=conf_min,
        iou=DETECT_IOU,
        imgsz=DETECT_IMGSZ,
        max_det=DETECT_MAX_DET,
        verbose=False,
    )
    detecciones = []

    for result, (_, offset_x, offset_y) in zip(results, tiles):
        names = result.names
        for box in result.boxes:
            det_conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y

            clase = None
            conf = det_conf

            if classifier is not None:
                clase_cls, conf_cls = clasificar_recorte(classifier, frame, x1, y1, x2, y2)
                if clase_cls is not None:
                    clase = clase_cls
                    conf = conf_cls
                    if conf < CLASSIFY_MINIMA and det_conf > conf:
                        conf = det_conf
            if clase is None:
                clase_id = int(box.cls.item())
                clase = names[clase_id]

            # --- FILTRO GEOMÉTRICO DE SEGURIDAD ---
            # Si el grano es muy largo, es casi imposible que sea "quebrado"
            # independientemente de lo que diga el clasificador visual.
            bw_val = x2 - x1
            bh_val = y2 - y1
            aspect_ratio = max(bw_val, bh_val) / max(1, min(bw_val, bh_val))
            
            # Ajuste de sensibilidad: 1.55 es un umbral más seguro para granos sanos
            if clase == "quebrado" and aspect_ratio > 1.55:
                # Si es largo, y la confianza de quebrado no es absoluta (>98%), lo pasamos a sano
                if conf < 0.98:
                    clase = "sano"
                    conf = max(conf, 0.85) 
            # --------------------------------------

            detecciones.append({"clase": clase, "conf": float(conf), "bbox": (x1, y1, x2, y2)})

    # --- REFINAMIENTO POR TAMAÑO COMPARATIVO ---
    if detecciones:
        # 1. Encontrar la longitud de referencia (el percentil 90 de los granos más largos)
        longitudes = []
        for d in detecciones:
            bx1, by1, bx2, by2 = d["bbox"]
            longitudes.append(max(bx2 - bx1, by2 - by1))
        
        # Usamos el percentil 90 para evitar ruidos pero captar el tamaño del grano sano promedio
        longitud_ref = np.percentile(longitudes, 90)
        
        for det in detecciones:
            bx1, by1, bx2, by2 = det["bbox"]
            longitud_actual = max(bx2 - bx1, by2 - by1)
            
            # Si la IA dice que es quebrado, pero mide más del 82% del largo de un sano... es sano.
            if det["clase"] == "quebrado" and longitud_actual > (longitud_ref * 0.82):
                det["clase"] = "sano"
                det["conf"] = max(det["conf"], 0.80)
    # -------------------------------------------

    return deduplicar_detecciones(detecciones, DETECT_IOU)


def resumir_detecciones(frame: np.ndarray, detecciones: list[dict]):
    conteos = {"sano": 0, "panza_blanca": 0, "quebrado": 0}
    confs_validas = []

    for det in detecciones:
        clase = det["clase"]
        conf = float(det["conf"])
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        dibujar_deteccion(frame, clase, conf, x1, y1, x2, y2)
        if clase in conteos:
            conteos[clase] += 1
            confs_validas.append(conf)

    return frame, conteos, confs_validas


def analizar_con_detector(models, frame: np.ndarray, conf_min: float):
    detecciones = predecir_detecciones(models, frame, conf_min)
    return resumir_detecciones(frame, detecciones)
