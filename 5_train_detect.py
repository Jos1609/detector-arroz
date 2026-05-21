"""
PASO 5: Entrenamiento YOLO de deteccion real
============================================

Entrena un detector usando fotos completas con varios granos y labels YOLO:
  data/etiquetado_manual/images
  data/etiquetado_manual/labels

Salida:
  modelo_arroz_detect.pt

Ejemplos:
  python 5_train_detect.py
  python 5_train_detect.py --epochs 80 --imgsz 960 --batch 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat
from multiprocessing.dummy import Pool as DummyThreadPool
from pathlib import Path

if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = str((Path(__file__).resolve().parent / "web" / "storage"))

from ultralytics import YOLO
import ultralytics.data.dataset as yolo_dataset
from ultralytics.data.dataset import DATASET_CACHE_VERSION
from ultralytics.data.utils import get_hash, save_dataset_cache_file, verify_image_label

yolo_dataset.ThreadPool = DummyThreadPool
yolo_dataset.NUM_THREADS = 1

SOURCE_IMAGES = Path("data/etiquetado_manual/images")
SOURCE_LABELS = Path("data/etiquetado_manual/labels")
DATASET_DIR = Path("data/detect_dataset")
DATASET_IMAGES = DATASET_DIR / "images"
DATASET_LABELS = DATASET_DIR / "labels"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
SUMMARY_PATH = DATASET_DIR / "summary.json"

CLASES = ["sano", "panza_blanca", "quebrado"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_VAL_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15
DEFAULT_EPOCHS = 80
DEFAULT_IMGSZ = 960
DEFAULT_BATCH = 4
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_RUN_NAME = "arroz_detector_ia"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ItemDataset:
    image_path: Path
    label_path: Path
    counts: dict[str, int]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def contar_clases(label_path: Path):
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


def cargar_items():
    if not SOURCE_IMAGES.exists() or not SOURCE_LABELS.exists():
        raise FileNotFoundError("No existe data/etiquetado_manual/images o labels")

    items = []
    for image_path in sorted(SOURCE_IMAGES.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTS:
            continue
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        counts = contar_clases(label_path)
        items.append(ItemDataset(image_path=image_path, label_path=label_path, counts=counts))

    if len(items) < 20:
        raise RuntimeError("Muy pocas imagenes para entrenar deteccion de forma confiable.")
    return items


def split_items(items: list[ItemDataset], val_size: float, test_size: float, seed: int):
    if val_size <= 0 or test_size < 0 or (val_size + test_size) >= 0.5:
        raise ValueError("Usa un val/test razonable. Ejemplo: val=0.15 test=0.15")

    rng = random.Random(seed)
    buckets = defaultdict(list)
    for item in items:
        firma = tuple(1 if item.counts.get(clase, 0) > 0 else 0 for clase in CLASES)
        buckets[firma].append(item)

    train_items = []
    val_items = []
    test_items = []

    for bucket in buckets.values():
        rng.shuffle(bucket)
        total_bucket = len(bucket)
        val_count = round(total_bucket * val_size)
        test_count = round(total_bucket * test_size)
        if total_bucket >= 3 and val_count == 0:
            val_count = 1
        if total_bucket >= 5 and test_count == 0:
            test_count = 1
        if val_count + test_count >= total_bucket:
            test_count = max(0, test_count - 1)

        train_cut = total_bucket - val_count - test_count
        train_items.extend(bucket[:train_cut])
        val_items.extend(bucket[train_cut:train_cut + val_count])
        test_items.extend(bucket[train_cut + val_count:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    train_count = len(train_items)
    if train_count < 10:
        raise RuntimeError("La particion deja muy pocas imagenes para train.")
    return {"train": train_items, "val": val_items, "test": test_items}


def copiar_split(split: str, items: list[ItemDataset]):
    (DATASET_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (DATASET_LABELS / split).mkdir(parents=True, exist_ok=True)

    for item in items:
        shutil.copy2(item.image_path, DATASET_IMAGES / split / item.image_path.name)
        destino_lbl = DATASET_LABELS / split / f"{item.image_path.stem}.txt"
        if item.label_path.exists():
            shutil.copy2(item.label_path, destino_lbl)
        else:
            destino_lbl.write_text("", encoding="utf-8")


def crear_cache_split(split: str):
    image_dir = DATASET_IMAGES / split
    label_dir = DATASET_LABELS / split
    im_files = sorted(str(p.resolve()) for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
    label_files = [str((label_dir / f"{Path(im).stem}.txt").resolve()) for im in im_files]
    cache_path = label_dir.with_suffix(".cache")

    x = {"labels": []}
    nm = nf = ne = nc = 0
    msgs = []
    iterable = zip(
        im_files,
        label_files,
        repeat(""),
        repeat(False),
        repeat(len(CLASES)),
        repeat(0),
        repeat(0),
        repeat(False),
    )

    for args in iterable:
        im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = verify_image_label(args)
        nm += nm_f
        nf += nf_f
        ne += ne_f
        nc += nc_f
        if im_file:
            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": shape,
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": segments,
                    "keypoints": keypoint,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if msg:
            msgs.append(msg)

    x["hash"] = get_hash(label_files + im_files)
    x["results"] = (nf, nm, ne, nc, len(im_files))
    x["msgs"] = msgs
    save_dataset_cache_file("", cache_path, x, DATASET_CACHE_VERSION)


def preparar_dataset(args):
    items = cargar_items()
    splits = split_items(items, val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split, split_items_list in splits.items():
        copiar_split(split, split_items_list)
        crear_cache_split(split)

    yaml = (
        f"path: {DATASET_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {CLASES}\n"
    )
    DATASET_YAML.write_text(yaml, encoding="utf-8")

    summary = {
        "classes": CLASES,
        "seed": args.seed,
        "splits": {},
    }
    for split, split_items_list in splits.items():
        agg = {clase: 0 for clase in CLASES}
        for item in split_items_list:
            for clase in CLASES:
                agg[clase] += item.counts.get(clase, 0)
        summary["splits"][split] = {
            "images": len(split_items_list),
            "boxes": int(sum(agg.values())),
            "per_class": agg,
        }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Dataset deteccion listo en {DATASET_DIR}")
    for split, info in summary["splits"].items():
        print(f"  {split:5s}: {info['images']:3d} imagenes | {info['boxes']:4d} cajas | {info['per_class']}")


def entrenar(args):
    print("=" * 60)
    print("  ENTRENAMIENTO YOLO DETECCION - GRANOS DE ARROZ")
    print("=" * 60)
    print(f"  Base model : {args.base_model}")
    print(f"  Epocas     : {args.epochs}")
    print(f"  Img size   : {args.imgsz}")
    print(f"  Batch      : {args.batch}")
    print(f"  Device     : {args.device}")
    print(f"  Run name   : {args.run_name}")
    print("=" * 60)

    model = YOLO(args.base_model)
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        cache=False,
        name=args.run_name,
        exist_ok=True,
        patience=max(20, args.epochs // 3),
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=10,
        degrees=4.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        mosaic=0.60,
        mixup=0.05,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        seed=args.seed,
        deterministic=True,
        verbose=True,
        plots=True,
    )

    posibles = sorted(Path(".").rglob(f"{args.run_name}/weights/best.pt"))
    if not posibles:
        raise RuntimeError("No se encontro best.pt del detector.")

    mejor_modelo = posibles[-1]
    shutil.copy2(mejor_modelo, "modelo_arroz_detect.pt")
    print(f"\nModelo detector guardado en modelo_arroz_detect.pt desde {mejor_modelo}")


def main():
    args = parse_args()
    preparar_dataset(args)
    if args.prepare_only:
        print("\nDataset preparado. Entrenamiento omitido por --prepare-only.")
        return
    entrenar(args)


if __name__ == "__main__":
    main()
