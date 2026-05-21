from __future__ import annotations

from flask import Blueprint, current_app, flash, redirect, request, send_file, url_for

from rice_app.db import get_db
from rice_app.services.auth_service import login_required
from rice_app.services.detection_service import allowed_file, process_sample_upload
from rice_app.services.lot_service import MINIMUM_SAMPLES, calculate_sample_percentages

bp = Blueprint("samples", __name__, url_prefix="/samples")


@bp.post("/create")
@login_required
def create():
    db = get_db()
    lot_id = request.form.get("lot_id", "").strip()
    image = request.files.get("image")

    if not lot_id:
        flash("No se encontro el lote seleccionado.", "error")
        return redirect(url_for("lots.index"))

    lot = db.execute("SELECT id, code FROM lots WHERE id = ?", (int(lot_id),)).fetchone()
    if lot is None:
        flash("El lote no existe.", "error")
        return redirect(url_for("lots.index"))

    if image is None or image.filename == "" or not allowed_file(image.filename):
        flash("Debes subir una imagen valida para procesar la muestra.", "error")
        return redirect(url_for("lots.detail", lot_id=lot["id"]))

    current_count = db.execute("SELECT COUNT(*) AS total FROM samples WHERE lot_id = ?", (lot["id"],)).fetchone()["total"]
    if current_count >= MINIMUM_SAMPLES:
        flash("Cada lote solo admite 3 pruebas. Ya no se pueden registrar mas muestras.", "error")
        return redirect(url_for("lots.detail", lot_id=lot["id"]))

    sample_number = current_count + 1

    try:
        processed = process_sample_upload(image, lot["code"], sample_number)
    except Exception as exc:
        flash(f"No se pudo procesar la imagen: {exc}", "error")
        return redirect(url_for("lots.detail", lot_id=lot["id"]))

    counts = processed["result"]["counts"]
    percentages = calculate_sample_percentages(counts)
    db.execute(
        """
        INSERT INTO samples (
            lot_id, sample_number, sample_weight_kg, image_filename, result_image_filename, result_json_filename,
            healthy_count, chalky_count, broken_count, total_detected, healthy_pct, chalky_pct, broken_pct, average_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            lot["id"],
            sample_number,
            0.0,
            processed["upload_name"],
            processed["result_image_name"],
            processed["result_json_name"],
            int(counts.get("sano", 0)),
            int(counts.get("panza_blanca", 0)),
            int(counts.get("quebrado", 0)),
            int(processed["result"]["total_detected"]),
            float(percentages["healthy_pct"]),
            float(percentages["chalky_pct"]),
            float(percentages["broken_pct"]),
            float(processed["result"]["average_confidence"]),
        ),
    )
    db.commit()
    flash(f"Prueba {sample_number} procesada y guardada correctamente.", "success")
    return redirect(url_for("lots.detail", lot_id=lot["id"]))


@bp.get("/files/<kind>/<filename>")
@login_required
def files(kind: str, filename: str):
    if kind == "uploads":
        base = current_app.config["UPLOADS_DIR"]
    else:
        base = current_app.config["RESULTS_DIR"]
    return send_file(base / filename)
