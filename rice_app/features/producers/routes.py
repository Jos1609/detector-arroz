from __future__ import annotations

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from rice_app.db import get_db
from rice_app.services.auth_service import login_required

bp = Blueprint("producers", __name__, url_prefix="/producers")


@bp.get("/")
@login_required
def index():
    producers = get_db().execute(
        """
        SELECT p.*, COUNT(l.id) AS lot_count
        FROM producers p
        LEFT JOIN lots l ON l.producer_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
        """
    ).fetchall()
    return render_template("producers/index.html", producers=producers)


@bp.get("/<int:producer_id>")
@login_required
def detail(producer_id: int):
    db = get_db()
    producer = db.execute(
        """
        SELECT *
        FROM producers
        WHERE id = ?
        """,
        (producer_id,),
    ).fetchone()
    if producer is None:
        abort(404)

    lots = db.execute(
        """
        SELECT l.id, l.code, l.variety, l.total_bags, l.created_at, COUNT(s.id) AS sample_count
        FROM lots l
        LEFT JOIN samples s ON s.lot_id = l.id
        WHERE l.producer_id = ?
        GROUP BY l.id
        ORDER BY l.created_at DESC
        """,
        (producer_id,),
    ).fetchall()

    return render_template("producers/detail.html", producer=producer, lots=lots)


@bp.route("/new", methods=["GET", "POST"])
@login_required
def create():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        document_number = request.form.get("document_number", "").strip()
        phone = request.form.get("phone", "").strip()
        location = request.form.get("location", "").strip()
        notes = request.form.get("notes", "").strip()

        if not name:
            flash("El nombre del productor es obligatorio.", "error")
        else:
            db = get_db()
            db.execute(
                """
                INSERT INTO producers (name, document_number, phone, location, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, document_number, phone, location, notes),
            )
            db.commit()
            flash("Productor registrado correctamente.", "success")
            return redirect(url_for("producers.index"))

    return render_template("producers/form.html")


@bp.route("/<int:producer_id>/delete", methods=["POST"])
@login_required
def delete(producer_id: int):
    db = get_db()
    
    producer = db.execute("SELECT id FROM producers WHERE id = ?", (producer_id,)).fetchone()
    if not producer:
        abort(404)
        
    try:
        db.execute(
            "DELETE FROM samples WHERE lot_id IN (SELECT id FROM lots WHERE producer_id = ?)",
            (producer_id,)
        )
        db.execute("DELETE FROM lots WHERE producer_id = ?", (producer_id,))
        db.execute("DELETE FROM producers WHERE id = ?", (producer_id,))
        db.commit()
        flash("Productor y sus lotes eliminados correctamente.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error al eliminar productor: {str(e)}", "error")
        
    return redirect(url_for("producers.index"))
