from __future__ import annotations

from flask import Blueprint, render_template

from rice_app.db import get_db
from rice_app.services.auth_service import login_required

bp = Blueprint("dashboard", __name__)


@bp.get("/")
@login_required
def index():
    db = get_db()
    stats = {
        "producers": db.execute("SELECT COUNT(*) AS total FROM producers").fetchone()["total"],
        "lots": db.execute("SELECT COUNT(*) AS total FROM lots").fetchone()["total"],
        "samples": db.execute("SELECT COUNT(*) AS total FROM samples").fetchone()["total"],
        "pending_minimum": db.execute(
            """
            SELECT COUNT(*) AS total
            FROM lots l
            LEFT JOIN samples s ON s.lot_id = l.id
            GROUP BY l.id
            HAVING COUNT(s.id) < 3
            """
        ).fetchall(),
    }

    recent_lots = db.execute(
        """
        SELECT l.id, l.code, l.variety, l.total_bags, p.name AS producer_name, COUNT(s.id) AS sample_count
        FROM lots l
        JOIN producers p ON p.id = l.producer_id
        LEFT JOIN samples s ON s.lot_id = l.id
        GROUP BY l.id
        ORDER BY l.created_at DESC
        LIMIT 8
        """
    ).fetchall()

    return render_template(
        "dashboard/index.html",
        stats={
            **stats,
            "pending_minimum": len(stats["pending_minimum"]),
        },
        recent_lots=recent_lots,
    )
