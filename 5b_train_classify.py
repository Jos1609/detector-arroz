import argparse
import os
import shutil
from pathlib import Path
import random
from ultralytics import YOLO

# Configuración de rutas
SOURCE_CROPS = Path("data/etiquetado_manual/crops")
DATASET_DIR = Path("data/classify_dataset")
MODEL_OUTPUT = "modelo_arroz_classify.pt"

# Clases esperadas
CLASES = ["sano", "panza_blanca", "quebrado"]

def preparar_dataset(val_split=0.2):
    print(f"--- Preparando dataset de clasificación en {DATASET_DIR} ---")
    
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        
    for split in ['train', 'val']:
        for clase in CLASES:
            (DATASET_DIR / split / clase).mkdir(parents=True, exist_ok=True)

    for clase in CLASES:
        clase_path = SOURCE_CROPS / clase
        if not clase_path.exists():
            print(f"Advertencia: No existe la carpeta de crops para {clase}")
            continue
            
        imagenes = list(clase_path.glob("*.jpg"))
        random.shuffle(imagenes)
        
        split_idx = int(len(imagenes) * (1 - val_split))
        train_ims = imagenes[:split_idx]
        val_ims = imagenes[split_idx:]
        
        print(f"  {clase}: {len(train_ims)} para train, {len(val_ims)} para val")
        
        for im in train_ims:
            shutil.copy2(im, DATASET_DIR / 'train' / clase / im.name)
        for im in val_ims:
            shutil.copy2(im, DATASET_DIR / 'val' / clase / im.name)

def entrenar(epochs=50, imgsz=224, batch=32):
    print("--- Iniciando entrenamiento de CLASIFICACIÓN ---")
    
    # Usamos el modelo base de clasificación de YOLOv8
    model = YOLO('yolov8n-cls.pt')
    
    results = model.train(
        data=str(DATASET_DIR.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/classify',
        name='arroz_classifier',
        exist_ok=True
    )
    
    # Mover el mejor modelo a la raíz usando búsqueda inteligente
    posibles = list(Path("runs/classify").rglob("best.pt"))
    if posibles:
        best_path = posibles[-1] # El más reciente
        shutil.copy2(best_path, MODEL_OUTPUT)
        print(f"--- Entrenamiento completado. Modelo guardado como {MODEL_OUTPUT} desde {best_path} ---")
    else:
        print("Error: No se encontró el modelo entrenado en la carpeta runs/classify.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    
    preparar_dataset()
    entrenar(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
