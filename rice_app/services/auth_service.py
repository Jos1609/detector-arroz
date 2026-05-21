from __future__ import annotations

from functools import wraps

from flask import redirect, session, url_for


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if not session.get("user_id"):
            return redirect(url_for("auth.login"))
        return view(**kwargs)

    return wrapped_view
