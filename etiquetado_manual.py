"""
Herramienta de etiquetado manual para granos mezclados.

Uso:
    python etiquetado_manual.py
    python etiquetado_manual.py --input-dir data/nuevas_fotos
"""

from __future__ import annotations

import argparse
import shutil
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from image_io import read_image
import detector_ia

CLASES = ["sano", "panza_blanca", "quebrado"]
COLORES = {
    "sano": "#5b8c5a",
    "panza_blanca": "#c9784a",
    "quebrado": "#8b8178",
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
TAM_RECORTE = 224
MIN_LADO_CAJA = 8
PADDING_RECORTE = 0.1
ZOOM_STEP = 1.25
ZOOM_MIN = 0.25
ZOOM_MAX = 8.0
APP_BG = "#f4eee2"
TOP_BG = "#264638"
SIDE_BG = "#e5d8c3"
SIDE_CARD = "#f8f2e8"
TEXT_MAIN = "#2f241d"
TEXT_SOFT = "#6a5a4d"
CANVAS_BG = "#d7ddcf"
BTN_PRIMARY = "#1f6f5f"
BTN_SECONDARY = "#a66a3f"
BTN_MUTED = "#7d8c73"


@dataclass
class Anotacion:
    clase: str
    x1: int
    y1: int
    x2: int
    y2: int


def redimensionar_cuadrado(img: np.ndarray, tam: int = TAM_RECORTE) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((tam, tam, 3), 255, dtype=np.uint8)

    escala = tam / max(h, w)
    nuevo_w = max(1, int(round(w * escala)))
    nuevo_h = max(1, int(round(h * escala)))
    redim = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)

    lienzo = np.full((tam, tam, 3), 255, dtype=np.uint8)
    y_off = (tam - nuevo_h) // 2
    x_off = (tam - nuevo_w) // 2
    lienzo[y_off:y_off + nuevo_h, x_off:x_off + nuevo_w] = redim
    return lienzo


def ordenar_puntos(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def recorte_rotado_desde_caja(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * PADDING_RECORTE))
    pad_y = int(round(bh * PADDING_RECORTE))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(w, x2 + pad_x)
    ry2 = min(h, y2 + pad_y)

    roi = img[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

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

    def puntaje(cnt):
        area = cv2.contourArea(cnt)
        if area <= 0:
            return -1e9
        x, y, cw, ch = cv2.boundingRect(cnt)
        centro = np.array([x + cw / 2.0, y + ch / 2.0], dtype=np.float32)
        dist = np.linalg.norm(centro - centro_roi)
        return area - dist * 4.0

    cnt = max(contornos, key=puntaje)
    if len(cnt) < 5:
        return roi

    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    if rw < 2 or rh < 2:
        return roi

    box = cv2.boxPoints(rect)
    box = ordenar_puntos(box)
    ancho = max(1, int(round(max(rw, rh))))
    alto = max(1, int(round(min(rw, rh))))
    destino = np.array([[0, 0], [ancho - 1, 0], [ancho - 1, alto - 1], [0, alto - 1]], dtype=np.float32)
    matriz = cv2.getPerspectiveTransform(box.astype(np.float32), destino)
    warp = cv2.warpPerspective(roi, matriz, (ancho, alto), borderValue=(255, 255, 255))
    if warp.shape[0] > warp.shape[1]:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    return warp


def listar_imagenes(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS])


class AppEtiquetado(tk.Tk):
    def __init__(self, input_dir: Path, output_dir: Path):
        super().__init__()
        self.title("Mesa de Etiquetado de Arroz")
        self.geometry("1380x860")
        self.minsize(1180, 760)
        self.configure(bg=APP_BG)

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.images_out = self.output_dir / "images"
        self.labels_out = self.output_dir / "labels"
        self.crops_out = self.output_dir / "crops"
        self.images_out.mkdir(parents=True, exist_ok=True)
        self.labels_out.mkdir(parents=True, exist_ok=True)
        self.crops_out.mkdir(parents=True, exist_ok=True)
        for clase in CLASES:
            (self.crops_out / clase).mkdir(parents=True, exist_ok=True)

        try:
            self.modelos_ia, self.ia_mode = detector_ia.cargar_modelo()
            print(f"Modelo IA cargado en modo: {self.ia_mode}")
        except Exception as e:
            print(f"No se pudo cargar el modelo IA: {e}")
            self.modelos_ia = None

        self.lista_imagenes: list[Path] = []
        self.total_en_carpeta = 0
        self.total_pendientes = 0
        self.indice_actual = 0
        self.clase_actual = CLASES[0]
        self.img_cv: np.ndarray | None = None
        self.img_tk = None
        self.ruta_actual: Path | None = None
        self.escala = 1.0
        self.escala_base = 1.0
        self.zoom_factor = 1.0
        self.anotaciones: list[Anotacion] = []
        self.cache_anotaciones: dict[Path, list[Anotacion]] = {}
        self.rect_temp = None
        self.start_x = None
        self.start_y = None
        self.cur_x = None
        self.cur_y = None
        self.seleccion_idx: int | None = None

        self._crear_ui()
        self._abrir_carpeta_inicial()
        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)

        self.bind("1", lambda e: self._cambiar_clase("sano"))
        self.bind("2", lambda e: self._cambiar_clase("panza_blanca"))
        self.bind("3", lambda e: self._cambiar_clase("quebrado"))
        self.bind("<Control-z>", lambda e: self._deshacer())
        self.bind("<Delete>", lambda e: self._deshacer())
        self.bind("<BackSpace>", lambda e: self._deshacer())
        self.bind("<Control-s>", lambda e: self._guardar_actual(mostrar_mensaje=True))
        self.bind("<space>", lambda e: self._siguiente())
        self.bind("<Left>", lambda e: self._anterior())
        self.bind("<Right>", lambda e: self._siguiente())
        self.bind("+", lambda e: self._ajustar_zoom(ZOOM_STEP))
        self.bind("-", lambda e: self._ajustar_zoom(1 / ZOOM_STEP))
        self.bind("0", lambda e: self._reset_zoom())
        self.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.bind("a", lambda e: self._auto_detectar())
        self.bind("A", lambda e: self._auto_detectar())
        self.bind("r", lambda e: self._rotar_imagen())
        self.bind("R", lambda e: self._rotar_imagen())

    def _crear_ui(self):
        top = tk.Frame(self, bg=TOP_BG, padx=28, pady=18)
        top.pack(fill="x")

        marca = tk.Frame(top, bg=TOP_BG)
        marca.pack(side="left")
        self.lbl_titulo = tk.Label(marca, text="Cuaderno de Arroz", font=("Georgia", 20, "bold"), bg=TOP_BG, fg="#f7f3eb")
        self.lbl_titulo.pack(anchor="w")
        tk.Label(marca, text="Etiquetado de lotes y revision visual", font=("Segoe UI", 10), bg=TOP_BG, fg="#d7e3d8").pack(anchor="w")

        estado_box = tk.Frame(top, bg="#355848", padx=16, pady=10)
        estado_box.pack(side="left", padx=24)
        self.lbl_estado = tk.Label(estado_box, text="", font=("Segoe UI", 10), bg="#355848", fg="#eef4ea")
        self.lbl_estado.pack(anchor="w")
        self.lbl_zoom = tk.Label(estado_box, text="Zoom 100%", font=("Segoe UI", 10, "bold"), bg="#355848", fg="#f0c36a")
        self.lbl_zoom.pack(anchor="w", pady=(4, 0))

        top_actions = tk.Frame(top, bg=TOP_BG)
        top_actions.pack(side="right")
        tk.Button(top_actions, text="📂 Abrir Carpeta", bg="#f0c36a", fg="#2f241d", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", command=self._seleccionar_carpeta).pack(side="left", padx=4)
        tk.Button(top_actions, text="Guardar hoja", bg=BTN_PRIMARY, fg="white", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", command=lambda: self._guardar_actual(mostrar_mensaje=True)).pack(side="left", padx=4)
        tk.Button(top_actions, text="Deshacer", bg=BTN_MUTED, fg="white", font=("Segoe UI", 10), relief="flat", cursor="hand2", command=self._deshacer).pack(side="left", padx=4)
        tk.Button(top_actions, text="🤖 Auto-detectar", bg="#4e9f3d", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", command=self._auto_detectar).pack(side="left", padx=4)
        tk.Button(top_actions, text="🔄 Rotar", bg=BTN_SECONDARY, fg="white", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", command=self._rotar_imagen).pack(side="left", padx=4)
        tk.Button(top_actions, text="⚙️", bg=TOP_BG, fg="#d7e3d8", font=("Segoe UI", 10), relief="flat", cursor="hand2", command=self._configurar_salida_manual).pack(side="left", padx=4)

        body = tk.Frame(self, bg=APP_BG)
        body.pack(fill="both", expand=True, padx=16, pady=(10, 12))

        centro = tk.Frame(body, bg=APP_BG)
        centro.pack(fill="both", expand=True)

        self.scroll_y = tk.Scrollbar(centro, orient="vertical")
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x = tk.Scrollbar(centro, orient="horizontal")
        self.scroll_x.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(centro, bg=CANVAS_BG, cursor="crosshair", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.scroll_x.config(command=self.canvas.xview)
        self.scroll_y.config(command=self.canvas.yview)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        bottom = tk.Frame(self, bg=SIDE_BG, padx=18, pady=14)
        bottom.pack(fill="x", padx=16, pady=(0, 16))

        left_summary = tk.Frame(bottom, bg=SIDE_BG)
        left_summary.pack(side="left", fill="both", expand=True)
        self.lbl_clase = tk.Label(left_summary, text="SANO", font=("Georgia", 18, "bold"), bg=SIDE_BG, fg=COLORES[self.clase_actual])
        self.lbl_clase.pack(anchor="w")
        self.lbl_resumen = tk.Label(left_summary, text="0 anotaciones", justify="left", font=("Segoe UI", 10, "bold"), bg=SIDE_BG, fg=TEXT_MAIN)
        self.lbl_resumen.pack(anchor="w", pady=(6, 0))
        self.lbl_carpeta = tk.Label(left_summary, text="Carpeta: -", justify="left", wraplength=460, font=("Segoe UI", 9), bg=SIDE_BG, fg=TEXT_SOFT)
        self.lbl_carpeta.pack(anchor="w", pady=(6, 0))

        center_classes = tk.Frame(bottom, bg=SIDE_BG)
        center_classes.pack(side="left", padx=18)
        tk.Label(center_classes, text="Seleccion rapida", font=("Segoe UI", 10, "bold"), bg=SIDE_BG, fg=TEXT_MAIN).pack(anchor="w", pady=(0, 8))
        self.btns_clase = {}
        chip_row = tk.Frame(center_classes, bg=SIDE_BG)
        chip_row.pack()
        for clase in CLASES:
            btn = tk.Button(chip_row, text=clase.upper(), bg=COLORES[clase], fg="#fffaf4", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", padx=12, pady=10, command=lambda c=clase: self._cambiar_clase(c))
            btn.pack(side="left", padx=4)
            self.btns_clase[clase] = btn

        right_tools = tk.Frame(bottom, bg=SIDE_BG)
        right_tools.pack(side="right")
        tk.Label(right_tools, text="Movimiento", font=("Segoe UI", 10, "bold"), bg=SIDE_BG, fg=TEXT_MAIN).pack(anchor="e", pady=(0, 8))
        row1 = tk.Frame(right_tools, bg=SIDE_BG)
        row1.pack(anchor="e")
        tk.Button(row1, text="Anterior", bg="#7f8f78", fg="white", font=("Segoe UI", 10), relief="flat", cursor="hand2", command=self._anterior).pack(side="left", padx=4)
        tk.Button(row1, text="Siguiente", bg=BTN_SECONDARY, fg="white", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", command=self._siguiente).pack(side="left", padx=4)
        row2 = tk.Frame(right_tools, bg=SIDE_BG)
        row2.pack(anchor="e", pady=(8, 0))
        tk.Button(row2, text="Zoom -", bg="#6d7b6c", fg="white", font=("Segoe UI", 10), relief="flat", cursor="hand2", command=lambda: self._ajustar_zoom(1 / ZOOM_STEP)).pack(side="left", padx=4)
        tk.Button(row2, text="Ajustar", bg="#d9c8a1", fg=TEXT_MAIN, font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", command=self._reset_zoom).pack(side="left", padx=4)
        tk.Button(row2, text="Zoom +", bg=BTN_SECONDARY, fg="white", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", command=lambda: self._ajustar_zoom(ZOOM_STEP)).pack(side="left", padx=4)

        list_wrap = tk.Frame(bottom, bg=SIDE_CARD, padx=10, pady=10)
        list_wrap.pack(side="right", padx=(18, 0))
        tk.Label(list_wrap, text="Registro", font=("Segoe UI", 10, "bold"), bg=SIDE_CARD, fg=TEXT_MAIN).pack(anchor="w")
        self.lst_anotaciones = tk.Listbox(list_wrap, bg=SIDE_CARD, fg=TEXT_MAIN, selectbackground="#b8ccb8", selectforeground=TEXT_MAIN, relief="flat", height=8, width=34, highlightthickness=0)
        self.lst_anotaciones.pack(fill="both", expand=True, pady=(8, 0))
        self.lst_anotaciones.bind("<<ListboxSelect>>", self._on_list_select)

    def _cambiar_clase(self, clase: str):
        self.clase_actual = clase
        self.lbl_clase.config(text=clase.upper(), fg=COLORES[clase])
        for nombre, btn in self.btns_clase.items():
            btn.config(relief="sunken" if nombre == clase else "flat", bd=2 if nombre == clase else 0)
        
        # Si hay algo seleccionado, cambiarle la clase
        if self.seleccion_idx is not None and 0 <= self.seleccion_idx < len(self.anotaciones):
            self.anotaciones[self.seleccion_idx].clase = clase
            if self.ruta_actual:
                self.cache_anotaciones[self.ruta_actual] = [Anotacion(**vars(a)) for a in self.anotaciones]
            self._redibujar()

    def _abrir_carpeta_inicial(self):
        if self._cargar_desde_carpeta(self.input_dir):
            return

    def _seleccionar_carpeta(self):
        carpeta = filedialog.askdirectory(title="Selecciona la carpeta con fotos para etiquetar", initialdir=str(self.input_dir if self.input_dir.exists() else Path.cwd()), mustexist=True)
        if not carpeta:
            return
        self._cargar_desde_carpeta(Path(carpeta))

    def _configurar_salida_manual(self):
        salida = filedialog.askdirectory(title="Selecciona donde guardar las etiquetas y recortes", initialdir=str(self.output_dir))
        if salida:
            self._configurar_salida(Path(salida))
            messagebox.showinfo("Configuración", f"Carpeta de salida cambiada a:\n{salida}")

    def _configurar_salida(self, path: Path):
        self.output_dir = path
        self.images_out = self.output_dir / "images"
        self.labels_out = self.output_dir / "labels"
        self.crops_out = self.output_dir / "crops"
        self.images_out.mkdir(parents=True, exist_ok=True)
        self.labels_out.mkdir(parents=True, exist_ok=True)
        self.crops_out.mkdir(parents=True, exist_ok=True)
        for clase in CLASES:
            (self.crops_out / clase).mkdir(parents=True, exist_ok=True)
        print(f"Nueva carpeta de salida configurada: {path}")

    def _cargar_desde_carpeta(self, carpeta: Path) -> bool:
        imagenes = listar_imagenes(carpeta)
        if not imagenes:
            messagebox.showwarning("Sin imagenes", f"No se encontraron imagenes en:\n{carpeta}")
            return False

        pendientes = [ruta for ruta in imagenes if not self._imagen_ya_etiquetada(carpeta, ruta)]
        
        self.input_dir = carpeta
        self.total_en_carpeta = len(imagenes)
        self.total_pendientes = len(pendientes)
        
        # Si no hay pendientes, preguntar si quiere revisar todas
        if not pendientes:
            if messagebox.askyesno("Carpeta terminada", "Todas las imagenes ya tienen etiquetas. ¿Deseas cargarlas todas para revisarlas?"):
                self.lista_imagenes = imagenes
            else:
                self.lbl_estado.config(text="No hay imagenes pendientes en esta carpeta")
                self.lbl_resumen.config(text="Todas las imagenes ya fueron etiquetadas")
                self.lst_anotaciones.delete(0, tk.END)
                self.canvas.delete("all")
                return True
        else:
            self.lista_imagenes = pendientes

        self.indice_actual = 0
        self.ruta_actual = None
        self.anotaciones = []
        self.lbl_carpeta.config(text=f"Carpeta:\n{self.input_dir}")

        self._cargar_imagen()
        return True

    def _cargar_imagen(self):
        if not self.lista_imagenes:
            return

        self.ruta_actual = self.lista_imagenes[self.indice_actual]
        self.img_cv = read_image(self.ruta_actual)
        if self.img_cv is None:
            messagebox.showwarning("Imagen invalida", f"No se pudo abrir:\n{self.ruta_actual}")
            self._mover_indice(1)
            return

        if self.ruta_actual in self.cache_anotaciones:
            self.anotaciones = [Anotacion(**vars(a)) for a in self.cache_anotaciones[self.ruta_actual]]
        else:
            self.anotaciones = self._cargar_anotaciones_guardadas(self.ruta_actual)
            self.cache_anotaciones[self.ruta_actual] = [Anotacion(**vars(a)) for a in self.anotaciones]

        self.zoom_factor = 1.0
        self._recalcular_escala_base()
        self._redibujar()
        self._actualizar_textos()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _recalcular_escala_base(self):
        if self.img_cv is None:
            return
        self.update_idletasks()
        max_w = max(500, self.canvas.winfo_width() - 10)
        max_h = max(400, self.canvas.winfo_height() - 10)
        h, w = self.img_cv.shape[:2]
        self.escala_base = min(max_w / w, max_h / h, 1.0)

    def _renderizar_fondo(self):
        self.escala = max(ZOOM_MIN, min(ZOOM_MAX, self.escala_base * self.zoom_factor))
        h, w = self.img_cv.shape[:2]
        new_w = max(1, int(round(w * self.escala)))
        new_h = max(1, int(round(h * self.escala)))
        rgb = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.lbl_zoom.config(text=f"Zoom {self.zoom_factor:.0%}")

    def _redibujar(self):
        if self.img_cv is None:
            return
        self.canvas.delete("all")
        self._renderizar_fondo()

        for idx, ann in enumerate(self.anotaciones):
            x1, y1, x2, y2 = self._to_canvas_coords(ann.x1, ann.y1, ann.x2, ann.y2)
            color = COLORES.get(ann.clase, "#ffffff")
            
            # Resaltar si está seleccionado
            es_seleccion = (self.seleccion_idx == idx)
            width = 4 if es_seleccion else 2
            
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
            
            if es_seleccion:
                # Dibujar un pequeño borde blanco extra para visibilidad
                self.canvas.create_rectangle(x1-1, y1-1, x2+1, y2+1, outline="white", width=1)

            self.canvas.create_rectangle(x1, max(0, y1 - 22), x1 + 150, y1, fill=color, outline=color)
            self.canvas.create_text(x1 + 4, max(0, y1 - 11), text=f"{idx+1}. {ann.clase}", anchor="w", fill="#111111", font=("Segoe UI", 9, "bold"))

        self._actualizar_lista()

    def _ajustar_zoom(self, factor: float):
        if self.img_cv is None:
            return
        min_factor = ZOOM_MIN / max(self.escala_base, 1e-6)
        max_factor = ZOOM_MAX / max(self.escala_base, 1e-6)
        self.zoom_factor = max(min_factor, min(max_factor, self.zoom_factor * factor))
        self._redibujar()

    def _reset_zoom(self):
        if self.img_cv is None:
            return
        self.zoom_factor = 1.0
        self._redibujar()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _on_ctrl_mousewheel(self, event):
        self._ajustar_zoom(ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP)

    def _actualizar_textos(self):
        total = len(self.lista_imagenes)
        actual = self.indice_actual + 1
        nombre = self.ruta_actual.name if self.ruta_actual else ""
        saltadas = self.total_en_carpeta - self.total_pendientes
        self.lbl_estado.config(text=f"Pendiente {actual} de {total} | {nombre} | Total carpeta: {self.total_en_carpeta} | Ya hechas: {saltadas}")
        self.lbl_resumen.config(text=f"{len(self.anotaciones)} anotaciones en esta imagen")

    def _actualizar_lista(self):
        self.lst_anotaciones.delete(0, tk.END)
        for idx, ann in enumerate(self.anotaciones, start=1):
            self.lst_anotaciones.insert(tk.END, f"{idx}. {ann.clase} | {ann.x2 - ann.x1}x{ann.y2 - ann.y1}")
        self._actualizar_textos()

    def _to_canvas_coords(self, x1: int, y1: int, x2: int, y2: int):
        return tuple(int(round(v * self.escala)) for v in (x1, y1, x2, y2))

    def _to_image_coords(self, x1: int, y1: int, x2: int, y2: int):
        if self.img_cv is None:
            return 0, 0, 0, 0
        h, w = self.img_cv.shape[:2]
        vals = [int(round(v / self.escala)) for v in (x1, y1, x2, y2)]
        ix1, iy1, ix2, iy2 = vals
        return max(0, min(w, ix1)), max(0, min(h, iy1)), max(0, min(w, ix2)), max(0, min(h, iy2))

    def _canvas_point(self, event):
        return int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))

    def _on_press(self, event):
        if self.img_tk is None:
            return
        cx, cy = self._canvas_point(event)
        
        # Intentar seleccionar una anotación existente primero
        # Usamos un pequeño margen de 5 píxeles para que sea más fácil hacer click
        margen_click = 5
        clicked_idx = None
        
        # Recorremos en orden inverso para seleccionar el que se ve "encima" si hay solapamiento
        for idx in range(len(self.anotaciones) - 1, -1, -1):
            ann = self.anotaciones[idx]
            x1, y1, x2, y2 = self._to_canvas_coords(ann.x1, ann.y1, ann.x2, ann.y2)
            if (x1 - margen_click) <= cx <= (x2 + margen_click) and \
               (y1 - margen_click) <= cy <= (y2 + margen_click):
                clicked_idx = idx
                break
        
        if clicked_idx is not None:
            self.seleccion_idx = clicked_idx
            self.lst_anotaciones.selection_clear(0, tk.END)
            self.lst_anotaciones.selection_set(clicked_idx)
            self.lst_anotaciones.see(clicked_idx)
            self._redibujar()
            
            # Guardamos el inicio por si en el futuro queremos permitir mover la caja
            self.start_x, self.start_y = cx, cy
            # Importante: No creamos rect_temp aquí, para que no se dibuje una caja nueva al hacer click en una existente
            return

        # Si hace click en el vacío, deseleccionamos y empezamos a dibujar una caja nueva
        self.seleccion_idx = None
        self._redibujar()
        
        self.start_x, self.start_y = cx, cy
        self.cur_x, self.cur_y = self.start_x, self.start_y
        if self.rect_temp is not None:
            self.canvas.delete(self.rect_temp)
        self.rect_temp = self.canvas.create_rectangle(self.start_x, self.start_y, self.cur_x, self.cur_y, outline=COLORES[self.clase_actual], width=2, dash=(4, 2))

    def _on_drag(self, event):
        if self.rect_temp is None:
            return
        self.cur_x, self.cur_y = self._canvas_point(event)
        self.canvas.coords(self.rect_temp, self.start_x, self.start_y, self.cur_x, self.cur_y)

    def _on_release(self, event):
        if self.rect_temp is None or self.img_cv is None:
            return

        self.cur_x, self.cur_y = self._canvas_point(event)
        x1, x2 = sorted((self.start_x, self.cur_x))
        y1, y2 = sorted((self.start_y, self.cur_y))
        if (x2 - x1) < MIN_LADO_CAJA or (y2 - y1) < MIN_LADO_CAJA:
            self.canvas.delete(self.rect_temp)
            self.rect_temp = None
            return

        ix1, iy1, ix2, iy2 = self._to_image_coords(x1, y1, x2, y2)
        if ix2 - ix1 < MIN_LADO_CAJA or iy2 - iy1 < MIN_LADO_CAJA:
            self.canvas.delete(self.rect_temp)
            self.rect_temp = None
            return

        self.anotaciones.append(Anotacion(self.clase_actual, ix1, iy1, ix2, iy2))
        self.canvas.delete(self.rect_temp)
        self.rect_temp = None
        self.seleccion_idx = len(self.anotaciones) - 1 # Seleccionar el nuevo
        self.cache_anotaciones[self.ruta_actual] = [Anotacion(**vars(a)) for a in self.anotaciones]
        self._redibujar()

    def _auto_detectar(self):
        if self.img_cv is None:
            return
        if self.modelos_ia is None:
            messagebox.showwarning("IA no disponible", "No se pudo cargar el modelo de detección.")
            return

        # Confirmar si ya hay anotaciones
        if self.anotaciones:
            if not messagebox.askyesno("Confirmar", "Ya existen anotaciones. ¿Deseas borrarlas y usar las de la IA?"):
                return

        self.anotaciones = []
        try:
            # Si el modelo IA es YOLO, espera BGR
            detecciones = detector_ia.predecir_detecciones(self.modelos_ia, self.img_cv, detector_ia.CONF_MINIMA)
            for det in detecciones:
                clase = det["clase"]
                x1, y1, x2, y2 = det["bbox"]
                self.anotaciones.append(Anotacion(clase, x1, y1, x2, y2))
            
            self.cache_anotaciones[self.ruta_actual] = [Anotacion(**vars(a)) for a in self.anotaciones]
            self._redibujar()
            messagebox.showinfo("Detección completa", f"Se detectaron {len(self.anotaciones)} granos.")
        except Exception as e:
            messagebox.showerror("Error de IA", f"Ocurrió un error durante la detección:\n{e}")

    def _rotar_imagen(self):
        if self.img_cv is None:
            return
        
        # Rotar imagen 90 grados sentido horario
        h, w = self.img_cv.shape[:2]
        self.img_cv = cv2.rotate(self.img_cv, cv2.ROTATE_90_CLOCKWISE)
        
        # Rotar anotaciones
        nuevas = []
        for ann in self.anotaciones:
            # Nueva x1 = h - vieja y2
            # Nueva y1 = vieja x1
            # Nueva x2 = h - vieja y1
            # Nueva y2 = vieja x2
            nx1 = h - ann.y2
            ny1 = ann.x1
            nx2 = h - ann.y1
            ny2 = ann.x2
            nuevas.append(Anotacion(ann.clase, nx1, ny1, nx2, ny2))
        
        self.anotaciones = nuevas
        if self.ruta_actual:
            self.cache_anotaciones[self.ruta_actual] = [Anotacion(**vars(a)) for a in self.anotaciones]
            
        self._recalcular_escala_base()
        self._redibujar()

    def _deshacer(self):
        if not self.anotaciones:
            return
        
        if self.seleccion_idx is not None:
            self.anotaciones.pop(self.seleccion_idx)
            self.seleccion_idx = None
        else:
            self.anotaciones.pop()
            
        if self.ruta_actual is not None:
            self.cache_anotaciones[self.ruta_actual] = [Anotacion(**vars(a)) for a in self.anotaciones]
        self._redibujar()

    def _on_list_select(self, event):
        selection = self.lst_anotaciones.curselection()
        if selection:
            self.seleccion_idx = selection[0]
            self._redibujar()

    def _nombre_base(self, ruta: Path) -> str:
        rel = ruta.relative_to(self.input_dir)
        partes = list(rel.parts)
        partes[-1] = Path(partes[-1]).stem
        return "__".join(partes)

    def _imagen_ya_etiquetada(self, base_dir: Path, ruta: Path) -> bool:
        rel = ruta.relative_to(base_dir)
        partes = list(rel.parts)
        partes[-1] = Path(partes[-1]).stem
        return (self.labels_out / f"{'__'.join(partes)}.txt").exists()

    def _cargar_anotaciones_guardadas(self, ruta: Path) -> list[Anotacion]:
        if self.img_cv is None:
            return []
        ruta_label = self.labels_out / f"{self._nombre_base(ruta)}.txt"
        if not ruta_label.exists():
            return []

        h, w = self.img_cv.shape[:2]
        anotaciones = []
        for linea in ruta_label.read_text(encoding="utf-8").splitlines():
            partes = linea.strip().split()
            if len(partes) != 5:
                continue
            try:
                clase_id = int(partes[0])
                xc, yc, bw, bh = map(float, partes[1:])
            except ValueError:
                continue
            if not 0 <= clase_id < len(CLASES):
                continue
            box_w = max(1, int(round(bw * w)))
            box_h = max(1, int(round(bh * h)))
            centro_x = int(round(xc * w))
            centro_y = int(round(yc * h))
            x1 = max(0, centro_x - box_w // 2)
            y1 = max(0, centro_y - box_h // 2)
            x2 = min(w, x1 + box_w)
            y2 = min(h, y1 + box_h)
            anotaciones.append(Anotacion(CLASES[clase_id], x1, y1, x2, y2))
        return anotaciones

    def _guardar_actual(self, mostrar_mensaje: bool = False):
        if self.ruta_actual is None or self.img_cv is None:
            return

        try:
            # 1. Preparar datos
            base = self._nombre_base(self.ruta_actual)
            h, w = self.img_cv.shape[:2]
            lineas = []
            
            for idx, ann in enumerate(self.anotaciones, start=1):
                clase_id = CLASES.index(ann.clase)
                xc = ((ann.x1 + ann.x2) / 2) / w
                yc = ((ann.y1 + ann.y2) / 2) / h
                bw = (ann.x2 - ann.x1) / w
                bh = (ann.y2 - ann.y1) / h
                lineas.append(f"{clase_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # 2. SOLO GUARDAR SI HAY ANOTACIONES (o si el usuario quiere guardar una imagen vacía explícitamente)
            # Para evitar archivos de 0 bytes por error de carga
            if not lineas and not self.anotaciones:
                # Si no hay nada, no sobreescribimos el archivo existente con basura
                return

            # 3. Escribir etiquetas primero (prioridad máxima)
            destino_lbl = self.labels_out / f"{base}.txt"
            destino_lbl.parent.mkdir(parents=True, exist_ok=True)
            destino_lbl.write_text("\n".join(lineas), encoding="utf-8")

            # 4. Copiar imagen original
            destino_img = self.images_out / f"{base}{self.ruta_actual.suffix.lower()}"
            destino_img.parent.mkdir(parents=True, exist_ok=True)
            if not destino_img.exists():
                shutil.copy2(self.ruta_actual, destino_img)

            # 5. Exportar recortes (limpiando viejos primero)
            for archivo in self.crops_out.rglob(f"{base}__*.jpg"):
                try: archivo.unlink()
                except: pass
                
            for idx, ann in enumerate(self.anotaciones, start=1):
                self._guardar_recorte(base, idx, ann)

            self.cache_anotaciones[self.ruta_actual] = [Anotacion(**vars(a)) for a in self.anotaciones]

            if mostrar_mensaje:
                messagebox.showinfo("Guardado Correcto", f"Se guardaron {len(lineas)} etiquetas en:\n{destino_lbl}")
        
        except Exception as e:
            messagebox.showerror("ERROR CRÍTICO AL GUARDAR", f"No se pudo guardar el trabajo de esta imagen:\n{e}")

    def _guardar_recorte(self, base: str, idx: int, ann: Anotacion):
        if self.img_cv is None:
            return
        recorte = recorte_rotado_desde_caja(self.img_cv, ann.x1, ann.y1, ann.x2, ann.y2)
        if recorte.size == 0:
            return
        cv2.imwrite(str(self.crops_out / ann.clase / f"{base}__{idx:03d}.jpg"), redimensionar_cuadrado(recorte))

    def _anterior(self):
        if self.indice_actual == 0:
            return
        self._guardar_actual()
        self._mover_indice(-1)

    def _siguiente(self):
        self._guardar_actual()
        if self.indice_actual >= len(self.lista_imagenes) - 1:
            messagebox.showinfo("Terminado", "Ya llegaste a la ultima imagen.\nLas etiquetas quedaron guardadas en la carpeta de salida.")
            return
        self._mover_indice(1)

    def _mover_indice(self, delta: int):
        nuevo = self.indice_actual + delta
        if 0 <= nuevo < len(self.lista_imagenes):
            self.indice_actual = nuevo
            self._cargar_imagen()

    def _al_cerrar(self):
        self._guardar_actual()
        self.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/imagenes", help="Carpeta con imagenes para etiquetar")
    parser.add_argument("--output-dir", default="data/etiquetado_manual", help="Carpeta donde se guardan labels YOLO y recortes exportados")
    args = parser.parse_args()

    app = AppEtiquetado(Path(args.input_dir), Path(args.output_dir))
    if app.winfo_exists():
        app.mainloop()


if __name__ == "__main__":
    main()
