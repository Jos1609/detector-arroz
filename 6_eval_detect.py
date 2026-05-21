"""
PASO 6: Evaluacion del detector
===============================

Mide el conteo por clase y el error total usando las imagenes etiquetadas.

Ejemplo:
  python 6_eval_detect.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import detect_web
import detector_ia

SOURCE_IMAGES = Path("data/etiquetado_manual/images")
SOURCE_LABELS = Path("data/etiquetado_manual/labels")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASES = ["sano", "panza_blanca", "quebrado"]
REPORT_PATH = Path("web/storage/eval_detect_report.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Maximo de imagenes a evaluar (0 = todas)")
    return parser.parse_args()


def leer_gt(label_path: Path):
    counts = {clase: 0 for clase in CLASES}
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        clase_id = int(parts[0])
        if 0 <= clase_id < len(CLASES):
            counts[CLASES[clase_id]] += 1
    return counts


def seleccionar_conteo(models, frame, image_path: Path, conf_min: float):
    detecciones_det = detector_ia.predecir_detecciones(models, frame.copy(), conf_min)
    _, conteos_det, confs = detector_ia.resumir_detecciones(frame.copy(), detecciones_det)
    promedio_det = float(sum(confs) / len(confs)) if confs else 0.0
    return conteos_det, promedio_det, "detect_only"


def main():
    args = parse_args()
    models, model_mode = detector_ia.cargar_modelo()
    conf_min = detector_ia.CONF_MINIMA

    total_images = 0
    exact_total = 0
    sum_abs_error = 0
    sum_signed_error = 0
    per_class_abs = {clase: 0 for clase in CLASES}
    mode_counts = {}
    peores = []

    for image_path in sorted(SOURCE_IMAGES.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTS:
            continue
        if args.limit and total_images >= args.limit:
            break
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        gt = leer_gt(label_path)
        gt_total = sum(gt.values())
        if gt_total <= 0:
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        pred, promedio, selected_mode = seleccionar_conteo(models, frame, image_path, conf_min)
        pred_total = sum(pred.values())
        abs_error = abs(pred_total - gt_total)

        total_images += 1
        exact_total += int(pred_total == gt_total)
        sum_abs_error += abs_error
        sum_signed_error += pred_total - gt_total
        mode_counts[selected_mode] = mode_counts.get(selected_mode, 0) + 1

        for clase in CLASES:
            per_class_abs[clase] += abs(pred[clase] - gt[clase])

        peores.append(
            {
                "image": image_path.name,
                "gt_total": gt_total,
                "pred_total": pred_total,
                "abs_error": abs_error,
                "gt": gt,
                "pred": pred,
                "avg_conf": round(promedio, 4),
                "mode": selected_mode,
            }
        )

    peores.sort(key=lambda item: item["abs_error"], reverse=True)
    report = {
        "model_mode": model_mode,
        "confidence": conf_min,
        "images_evaluated": total_images,
        "count_mae": round(sum_abs_error / total_images, 4) if total_images else None,
        "count_bias": round(sum_signed_error / total_images, 4) if total_images else None,
        "exact_count_rate": round(exact_total / total_images, 4) if total_images else None,
        "per_class_mae": {
            clase: round(per_class_abs[clase] / total_images, 4) if total_images else None
            for clase in CLASES
        },
        "mode_usage": mode_counts,
        "worst_cases": peores[:15],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
