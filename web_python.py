"""
Punto de entrada del sistema web de arroz.

Uso:
    python web_python.py
"""

import threading
import webbrowser

from rice_app import create_app

app = create_app()
APP_HOST = "127.0.0.1"
APP_PORT = 8001


def open_browser():
    webbrowser.open(f"http://{APP_HOST}:{APP_PORT}", new=1)


if __name__ == "__main__":
    threading.Timer(1.2, open_browser).start()
    app.run(host=APP_HOST, port=APP_PORT, debug=False, use_reloader=False)
