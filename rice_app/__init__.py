from __future__ import annotations

import os

from flask import Flask, redirect, session, url_for

from rice_app.config import Config, WEB_STORAGE_DIR
from rice_app.db import init_app as init_db_app
from rice_app.db import init_db
from rice_app.features.auth.routes import bp as auth_bp
from rice_app.features.dashboard.routes import bp as dashboard_bp
from rice_app.features.lots.routes import bp as lots_bp
from rice_app.features.producers.routes import bp as producers_bp
from rice_app.features.samples.routes import bp as samples_bp
from rice_app.services.detection_service import ensure_storage_dirs


if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = str(WEB_STORAGE_DIR.resolve())


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    init_db_app(app)

    with app.app_context():
        ensure_storage_dirs()
        init_db()

    @app.before_request
    def enforce_auth():
        from flask import request

        allowed_endpoints = {"auth.login", "static"}
        if session.get("user_id"):
            return None
        if request.endpoint not in allowed_endpoints and request.endpoint != "auth.logout":
            return redirect(url_for("auth.login"))
        return None

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(producers_bp)
    app.register_blueprint(lots_bp)
    app.register_blueprint(samples_bp)

    return app
