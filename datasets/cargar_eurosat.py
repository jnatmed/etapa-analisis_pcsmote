# codigo/datasets/cargar_eurosat.py
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def cargar_dataset_eurosat(path, size=(64, 64)):
    path = Path(path)
    clases = sorted([d.name for d in path.iterdir() if d.is_dir()])
    X = []
    y = []

    for clase in clases:
        for img_path in (path / clase).glob("*.jpg"):
            img = Image.open(img_path).convert("RGB").resize(size)
            X.append(np.array(img))
            y.append(clase)

    X = np.stack(X)
    y = LabelEncoder().fit_transform(y)
    return X, y, clases
