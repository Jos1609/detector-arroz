from __future__ import annotations

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from rice_app.db import get_db
from rice_app.services.auth_service import login_required
from rice_app.services.lot_service import summarize_lot

bp = Blueprint("lots", __name__, url_prefix="/lots")


def _generate_next_lot_code(db) -> str:
    latest_numeric_code = db.execute(
        """
        SELECT code
        FROM lots
        WHERE code <> '' AND code NOT GLOB '*[^0-9]*'
        ORDER BY CAST(code AS INTEGER) DESC
        LIMIT 1
        """
    ).fetchone()
    next_number = int(latest_numeric_code["code"]) + 1 if latest_numeric_code else 1
    return f"{next_number:06d}"


def _get_lot_or_404(lot_id: int):
    lot = get_db().execute(
        """
        SELECT l.*, p.name AS producer_name
        FROM lots l
        JOIN producers p ON p.id = l.producer_id
        WHERE l.id = ?
        """,
        (lot_id,),
    ).fetchone()
    if lot is None:
        abort(404)
    return lot


@bp.get("/")
@login_required
def index():
    lots = get_db().execute(
        """
        SELECT l.id, l.code, l.variety, l.total_bags, l.created_at, p.name AS producer_name, COUNT(s.id) AS sample_count
        FROM lots l
        JOIN producers p ON p.id = l.producer_id
        LEFT JOIN samples s ON s.lot_id = l.id
        GROUP BY l.id
        ORDER BY l.created_at DESC
        """
    ).fetchall()
    return render_template("lots/index.html", lots=lots)


@bp.route("/new", methods=["GET", "POST"])
@login_required
def create():
    db = get_db()
    producers = db.execute("SELECT id, name FROM producers ORDER BY name ASC").fetchall()
    selected_producer_id = request.args.get("producer_id", "").strip()
    next_lot_code = _generate_next_lot_code(db)

    if request.method == "POST":
        producer_id = request.form.get("producer_id", "").strip()
        variety = request.form.get("variety", "").strip()
        total_bags = request.form.get("total_bags", "").strip()
        notes = request.form.get("notes", "").strip()

        error = None
        if not producer_id:
            error = "Debes seleccionar un productor."
        elif not variety:
            error = "La variedad es obligatoria."
        elif not total_bags:
            error = "La cantidad de sacos del lote es obligatoria."

        if error:
            flash(error, "error")
        else:
            try:
                total_bags_value = float(total_bags)
                if total_bags_value <= 0:
                    raise ValueError
                code = _generate_next_lot_code(db)
                db.execute(
                    """
                    INSERT INTO lots (producer_id, code, variety, total_bags, notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (int(producer_id), code, variety, total_bags_value, notes),
                )
                db.commit()
                flash("Lote registrado correctamente.", "success")
                new_lot_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
                return redirect(url_for("lots.detail", lot_id=new_lot_id))
            except ValueError:
                flash("La cantidad de sacos del lote debe ser numerica y mayor que cero.", "error")
            except Exception as e:
                db.rollback()
                flash(f"No se pudo registrar el lote. Error: {str(e)}", "error")

    return render_template(
        "lots/form.html",
        producers=producers,
        selected_producer_id=selected_producer_id,
        next_lot_code=next_lot_code,
    )


@bp.get("/<int:lot_id>")
@login_required
def detail(lot_id: int):
    db = get_db()
    lot = _get_lot_or_404(lot_id)
    samples = db.execute(
        """
        SELECT *
        FROM samples
        WHERE lot_id = ?
        ORDER BY sample_number ASC
        """,
        (lot_id,),
    ).fetchall()
    summary = summarize_lot(lot["total_bags"], samples)
    next_sample_number = len(samples) + 1
    return render_template(
        "lots/detail.html",
        lot=lot,
        samples=samples,
        summary=summary,
        next_sample_number=next_sample_number,
    )


@bp.route("/<int:lot_id>/delete", methods=["POST"])
@login_required
def delete(lot_id: int):
    db = get_db()
    lot = _get_lot_or_404(lot_id)
    producer_id = lot["producer_id"]
    try:
        db.execute("DELETE FROM samples WHERE lot_id = ?", (lot_id,))
        db.execute("DELETE FROM lots WHERE id = ?", (lot_id,))
        db.commit()
        flash("Lote eliminado correctamente.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error al eliminar lote: {str(e)}", "error")
        
    return redirect(url_for("producers.detail", producer_id=producer_id))
