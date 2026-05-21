"""
Herramienta de recorte manual de granos.

Uso:
    py 1b_crop_manual.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from image_io import read_image

INPUT_DIR = Path("data/imagenes")
OUTPUT_DIR = Path("data/imagenes_crop")
CLASES = ["sano", "panza_blanca", "quebrado"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
TAM_SALIDA = 224
ZOOM_STEP = 1.25
ZOOM_MIN = 0.25
ZOOM_MAX = 8.0
APP_BG = "#efe6d8"
TOP_BG = "#3b4f3f"
CANVAS_BG = "#d9e0d0"
TEXT_LIGHT = "#f8f4ed"
TEXT_SOFT = "#d6e0d3"
BTN_PRIMARY = "#1f6f5f"
BTN_SECONDARY = "#c9784a"
BTN_MUTED = "#7e8d77"
BTN_ACCENT = "#e0c27a"


def redimensionar_cuadrado(img: np.ndarray, tam: int = TAM_SALIDA) -> np.ndarray:
    h, w = img.shape[:2]
    escala = tam / max(h, w)
    nuevo_w = max(1, int(round(w * escala)))
    nuevo_h = max(1, int(round(h * escala)))
    redim = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)

    lienzo = np.full((tam, tam, 3), 255, dtype=np.uint8)
    y_off = (tam - nuevo_h) // 2
    x_off = (tam - nuevo_w) // 2
    lienzo[y_off:y_off + nuevo_h, x_off:x_off + nuevo_w] = redim
    return lienzo


class AppRecorte(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mesa de Recorte de Arroz")
        self.geometry("1100x760")
        self.configure(bg=APP_BG)

        self.lista_imagenes: list[tuple[str, Path]] = []
        for clase in CLASES:
            carpeta = INPUT_DIR / clase
            if carpeta.exists():
                for archivo in sorted(carpeta.glob("*")):
                    if archivo.is_file() and archivo.suffix.lower() in EXTS:
                        self.lista_imagenes.append((clase, archivo))

        if not self.lista_imagenes:
            messagebox.showerror("Error", f"No se encontraron imagenes en {INPUT_DIR}")
            self.destroy()
            return

        self.indice_actual = 0
        self.img_cv: np.ndarray | None = None
        self.img_tk = None
        self.escala = 1.0
        self.escala_base = 1.0
        self.zoom_factor = 1.0
        self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.cur_x = None
        self.cur_y = None

        self._crear_ui()
        self._cargar_imagen()

        self.bind("<Return>", lambda e: self._guardar_recorte())
        self.bind("<space>", lambda e: self._siguiente())
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("+", lambda e: self._ajustar_zoom(ZOOM_STEP))
        self.bind("-", lambda e: self._ajustar_zoom(1 / ZOOM_STEP))
        self.bind("0", lambda e: self._reset_zoom())
        self.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)

    def _crear_ui(self):
        top = tk.Frame(self, bg=TOP_BG, padx=24, pady=16)
        top.pack(fill="x")

        titulo = tk.Frame(top, bg=TOP_BG)
        titulo.pack(side="left")
        tk.Label(titulo, text="Tablero de Recorte", font=("Georgia", 18, "bold"), bg=TOP_BG, fg=TEXT_LIGHT).pack(anchor="w")
        tk.Label(titulo, text="Preparacion de muestras para entrenamiento", font=("Segoe UI", 10), bg=TOP_BG, fg=TEXT_SOFT).pack(anchor="w")

        estado = tk.Frame(top, bg="#526654", padx=14, pady=10)
        estado.pack(side="left", padx=18)
        self.lbl_info = tk.Label(estado, text="...", font=("Segoe UI", 10, "bold"), bg="#526654", fg=TEXT_LIGHT)
        self.lbl_info.pack(anchor="w")
        self.lbl_zoom = tk.Label(estado, text="Zoom 100%", font=("Segoe UI", 10, "bold"), bg="#526654", fg=BTN_ACCENT)
        self.lbl_zoom.pack(anchor="w", pady=(4, 0))

        acciones = tk.Frame(top, bg=TOP_BG)
        acciones.pack(side="right")
        tk.Button(acciones, text="Guardar", command=self._guardar_recorte, bg=BTN_PRIMARY, fg="white", font=("Segoe UI", 10, "bold"), relief="flat").pack(side="left", padx=4)
        tk.Button(acciones, text="Saltar", command=self._siguiente, bg=BTN_MUTED, fg="white", font=("Segoe UI", 10), relief="flat").pack(side="left", padx=4)
        tk.Button(acciones, text="Zoom -", command=lambda: self._ajustar_zoom(1 / ZOOM_STEP), bg=BTN_MUTED, fg="white", font=("Segoe UI", 10), relief="flat").pack(side="left", padx=4)
        tk.Button(acciones, text="Ajustar", command=self._reset_zoom, bg=BTN_ACCENT, fg="#2f241d", font=("Segoe UI", 10, "bold"), relief="flat").pack(side="left", padx=4)
        tk.Button(acciones, text="Zoom +", command=lambda: self._ajustar_zoom(ZOOM_STEP), bg=BTN_SECONDARY, fg="white", font=("Segoe UI", 10, "bold"), relief="flat").pack(side="left", padx=4)

        ayuda = tk.Frame(self, bg=APP_BG, padx=20, pady=8)
        ayuda.pack(fill="x")
        tk.Label(
            ayuda,
            text="Atajos: Enter guarda, Espacio salta, +/- cambia zoom, 0 reajusta, Ctrl+Rueda acerca o aleja.",
            font=("Segoe UI", 10),
            bg=APP_BG,
            fg="#5f6a5f",
        ).pack(anchor="w")

        frame = tk.Frame(self, bg=APP_BG)
        frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.scroll_y = tk.Scrollbar(frame, orient="vertical")
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x = tk.Scrollbar(frame, orient="horizontal")
        self.scroll_x.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(frame, bg=CANVAS_BG, cursor="crosshair", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.scroll_x.config(command=self.canvas.xview)
        self.scroll_y.config(command=self.canvas.yview)

        self.canvas.bind("<ButtonPress-1>", self._on_clic_izq)
        self.canvas.bind("<B1-Motion>", self._on_arrastrar)
        self.canvas.bind("<ButtonRelease-1>", self._on_soltar_clic)

    def _cargar_imagen(self):
        if self.indice_actual >= len(self.lista_imagenes):
            messagebox.showinfo("Terminado", "Has revisado todas las imagenes.")
            self.destroy()
            return

        clase, ruta = self.lista_imagenes[self.indice_actual]
        self.lbl_info.config(text=f"{clase.upper()} | Imagen {self.indice_actual + 1} de {len(self.lista_imagenes)} | {ruta.name}")

        self.img_cv = read_image(ruta)
        if self.img_cv is None:
            self._siguiente()
            return

        self.zoom_factor = 1.0
        self._recalcular_escala_base()
        self._renderizar_imagen()

        self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.cur_x = None
        self.cur_y = None
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _recalcular_escala_base(self):
        if self.img_cv is None:
            return
        alto, ancho = self.img_cv.shape[:2]
        self.escala_base = min(900 / ancho, 600 / alto, 1.0)

    def _renderizar_imagen(self):
        if self.img_cv is None:
            return
        self.escala = max(ZOOM_MIN, min(ZOOM_MAX, self.escala_base * self.zoom_factor))
        alto, ancho = self.img_cv.shape[:2]
        nuevo_w = max(1, int(round(ancho * self.escala)))
        nuevo_h = max(1, int(round(alto * self.escala)))

        rgb = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((nuevo_w, nuevo_h), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
        self.canvas.config(scrollregion=(0, 0, nuevo_w, nuevo_h))
        self.lbl_zoom.config(text=f"Zoom {self.zoom_factor:.0%}")

    def _ajustar_zoom(self, factor: float):
        if self.img_cv is None:
            return
        min_factor = ZOOM_MIN / max(self.escala_base, 1e-6)
        max_factor = ZOOM_MAX / max(self.escala_base, 1e-6)
        self.zoom_factor = max(min_factor, min(max_factor, self.zoom_factor * factor))
        self._renderizar_imagen()

    def _reset_zoom(self):
        if self.img_cv is None:
            return
        self.zoom_factor = 1.0
        self._renderizar_imagen()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _on_ctrl_mousewheel(self, event):
        self._ajustar_zoom(ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP)

    def _canvas_point(self, event):
        return int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))

    def _on_clic_izq(self, event):
        self.start_x, self.start_y = self._canvas_point(event)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="#00ff00", width=2)

    def _on_arrastrar(self, event):
        self.cur_x, self.cur_y = self._canvas_point(event)
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, self.cur_x, self.cur_y)

    def _on_soltar_clic(self, event):
        self.cur_x, self.cur_y = self._canvas_point(event)

    def _guardar_recorte(self):
        clase, ruta = self.lista_imagenes[self.indice_actual]
        if not self.rect_id or self.start_x is None or self.cur_x is None:
            messagebox.showwarning("Advertencia", "Dibuja un rectangulo primero sobre el grano.")
            return

        x1, x2 = min(self.start_x, self.cur_x), max(self.start_x, self.cur_x)
        y1, y2 = min(self.start_y, self.cur_y), max(self.start_y, self.cur_y)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return

        ox1, ox2 = int(round(x1 / self.escala)), int(round(x2 / self.escala))
        oy1, oy2 = int(round(y1 / self.escala)), int(round(y2 / self.escala))

        w, h = ox2 - ox1, oy2 - oy1
        px, py = int(w * 0.10), int(h * 0.10)
        alto, ancho = self.img_cv.shape[:2]
        fx1, fx2 = max(0, ox1 - px), min(ancho, ox2 + px)
        fy1, fy2 = max(0, oy1 - py), min(alto, oy2 + py)

        recorte = self.img_cv[fy1:fy2, fx1:fx2]
        carpeta = OUTPUT_DIR / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(carpeta / f"{ruta.stem}.jpg"), redimensionar_cuadrado(recorte))

        self.canvas.create_rectangle(x1, y1, x2, y2, outline="white", width=4, fill="white", stipple="gray25")
        self.after(60, self._siguiente)

    def _siguiente(self):
        self.indice_actual += 1
        self._cargar_imagen()


if __name__ == "__main__":
    app = AppRecorte()
    app.mainloop()
