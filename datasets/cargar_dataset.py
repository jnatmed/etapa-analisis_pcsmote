import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from cargar_eurosat import cargar_dataset_eurosat
from esquemas_conocidos import ESQUEMAS_CONOCIDOS


def cargar_dataset(path, clase_minoria=None, col_features=None, col_target=None,
                   sep=' ', header=None, binarizar=True, tipo='tabular',
                   impute='median', na_values=('?', 'NA', 'None'),
                   dataset_name=None, names=None):
    """
    Carga un dataset tabular o de imágenes y devuelve (X, y, clases).

    Parámetros clave:
    - tipo: 'tabular' | 'imagen' | 'tabular_npz'
    - header: None si el archivo NO tiene encabezado; 0 si la primera fila es encabezado.
    - names: lista completa de nombres de columnas para read_csv cuando header=None.
    - dataset_name: si header=None y names no fue provisto, se intentará usar ESQUEMAS_CONOCIDOS[dataset_name].
    - col_features: lista con los nombres de columnas de features (solo para 'tabular').
    - col_target: nombre (str) o lista de nombres de la(s) columna(s) objetivo (solo para 'tabular').
    - impute: 'median' o 'drop' (solo para 'tabular').
    - binarizar: si True, y exige 'clase_minoria'.

    Retorna:
    - df_features (pd.DataFrame con float32)
    - y (np.ndarray 1D)
    - clases (np.ndarray con etiquetas únicas o [0,1] si binariza)
    """
    # --- Imágenes ---
    # if tipo == 'imagen':
    #     X, y, clases = cargar_dataset_eurosat(path)
    #     return X, y, clases

    # --- TABULAR NPZ (US CRIME, etc.) ---
    if tipo == 'tabular_npz':
        data_npz = np.load(path)

        # Soportar dos formatos típicos:
        # 1) 'X' / 'y'
        # 2) 'data' / 'label'  ← el tuyo (US Crime)
        if 'X' in data_npz and 'y' in data_npz:
            X = data_npz['X']
            y = data_npz['y']
        elif 'data' in data_npz and 'label' in data_npz:
            X = data_npz['data']
            y = data_npz['label']
        else:
            raise KeyError(
                f"El archivo NPZ {path} debe contener ('X','y') o ('data','label'). "
                f"Claves encontradas: {data_npz.files}"
            )

        # Asegurar vector 1D
        y = np.asarray(y).ravel()
        X = np.asarray(X)

        # Definir nombres de columnas si no vienen dados
        n_features = X.shape[1]
        if col_features is None:
            col_features = [f"feat_{i}" for i in range(n_features)]

        # Construir DataFrame de features
        df_features = pd.DataFrame(X, columns=col_features)
        df_features = df_features.astype('float32')

        # Chequeo de NaN/inf en X
        matriz = df_features.to_numpy()
        if not np.isfinite(matriz).all():
            raise ValueError("❌ X (NPZ) contiene NaN o infinitos luego del preprocesamiento.")

        # Binarización (si corresponde)
        if binarizar:
            if clase_minoria is None:
                raise ValueError("Debe indicarse clase_minoria si se va a binarizar.")
            y = np.where(y == clase_minoria, 1, 0).astype(int)
            clases = np.array([0, 1])
        else:
            clases = np.unique(y)

        return df_features, y, clases

    # ───────────────────────── TABULAR (CSV/ESPACIOS) ─────────────────────────

    # --- Validaciones mínimas para tabular clásico ---
    if col_target is None or col_features is None:
        raise ValueError("Debés especificar col_features y col_target para tipo='tabular'.")

    # --- Selección de esquema/nombres para read_csv ---
    usar_names = None
    if names is not None:
        usar_names = list(names)
    elif header is None and dataset_name is not None and dataset_name in ESQUEMAS_CONOCIDOS:
        usar_names = ESQUEMAS_CONOCIDOS[dataset_name]

    # --- Lectura robusta ---
    if usar_names is not None:
        df = pd.read_csv(
            path,
            header=None,
            names=usar_names,
            sep=sep,
            na_values=list(na_values),
            engine='python',
            skipinitialspace=True
        )
    else:
        df = pd.read_csv(
            path,
            header=header,
            sep=sep,
            na_values=list(na_values),
            engine='python',
            skipinitialspace=True
        )

    # --- Anti-encabezado duplicado (si se pasó names y header=None) ---
    if usar_names is not None and header is None and len(df) > 0:
        fila0 = df.iloc[0].astype(str).str.strip().tolist()
        esquema = [str(c).strip() for c in usar_names]
        coincidencias = 0
        i = 0
        limite = min(len(fila0), len(esquema))
        while i < limite:
            if fila0[i] == esquema[i]:
                coincidencias += 1
            i += 1
        umbral = max(3, int(0.6 * limite))
        if coincidencias >= umbral:
            df = df.iloc[1:].reset_index(drop=True)

    # --- Mapeo automático por dataset_name si no se usó names y coincide cantidad ---
    if header is None and usar_names is None and dataset_name in ESQUEMAS_CONOCIDOS:
        esquema = ESQUEMAS_CONOCIDOS[dataset_name]
        if len(esquema) == df.shape[1]:
            df.columns = esquema

    # --- Validación de columnas requeridas (evita KeyError crípticos) ---
    columnas_requeridas = []
    for c in col_features:
        columnas_requeridas.append(c)
    if isinstance(col_target, str):
        columnas_requeridas.append(col_target)
    else:
        for c in col_target:
            columnas_requeridas.append(c)

    cols_faltantes = []
    for c in columnas_requeridas:
        if c not in df.columns:
            cols_faltantes.append(c)

    if len(cols_faltantes) > 0:
        primeras = df.columns.tolist()
        primeras = primeras[:min(20, len(primeras))]
        raise KeyError(
            f"Columnas ausentes: {cols_faltantes}. "
            f"Leídas={len(df.columns)} → {primeras}..."
        )

    # --- Selección de features y target ---
    df_features = df[col_features].apply(pd.to_numeric, errors='coerce')

    # --- Saneamiento numérico crítico ---
    # 1) Convertir infinitos en NaN (fillna no los arregla)
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # 2) Detectar columnas completamente inválidas (todo NaN)
    cols_todo_nan = df_features.columns[df_features.isna().all(axis=0)].tolist()
    if len(cols_todo_nan) > 0:
        raise ValueError(
            f"❌ Features con todos NaN tras to_numeric(): {cols_todo_nan}. "
            f"Revisá separador/decimales/valores no numéricos en esas columnas."
        )


    if isinstance(col_target, str):
        df_target = df[[col_target]]
    else:
        df_target = df[col_target]

    # --- Imputación / limpieza ---
    if impute == 'drop':
        mask_valid = df_features.notna().all(axis=1) & df_target.notna().all(axis=1)
        df_features = df_features.loc[mask_valid]
        df_target = df_target.loc[mask_valid]
    elif impute == 'median':
        med = df_features.median()  # ya es numérico
        if med.isna().any():
            cols_med_nan = med[med.isna()].index.tolist()
            raise ValueError(
                f"❌ No se pudo imputar por mediana: columnas sin mediana (todo NaN): {cols_med_nan}"
            )
        df_features = df_features.fillna(med)

        df_target = df_target.dropna(axis=0)
        df_features = df_features.loc[df_target.index]

    else:
        raise ValueError("impute debe ser 'median' o 'drop'.")

    # --- Tipos y chequeo de finitos ---
    df_features = df_features.astype('float32')
    matriz = df_features.to_numpy()
    if not np.isfinite(matriz).all():
        raise ValueError("❌ X contiene NaN o infinitos luego del preprocesamiento.")

    # --- Vector objetivo ---
    y = df_target.values.ravel()

    # --- Binarización (si corresponde) ---
    if binarizar:
        if clase_minoria is None:
            raise ValueError("Debe indicarse clase_minoria si se va a binarizar.")
        y = np.where(y == clase_minoria, 1, 0).astype(int)
        clases = np.array([0, 1])
    else:
        clases = pd.Series(y).unique()

    # --- Etiquetas de columnas en X (por claridad) ---
    if isinstance(col_features[0], str):
        df_features.columns = col_features

    return df_features, y, clases


def graficar_distribucion_clases(y, nombre_dataset, clases_labels=None, guardar_en=None):
    conteo_dict = Counter(y)
    clases = []
    cantidades = []
    for k, v in conteo_dict.items():
        clases.append(k)
        cantidades.append(v)

    if clases_labels:
        clases_mapeadas = []
        i = 0
        while i < len(clases):
            c = clases[i]
            if c in clases_labels:
                clases_mapeadas.append(clases_labels[c])
            else:
                clases_mapeadas.append(c)
            i += 1
        clases = clases_mapeadas

    plt.figure(figsize=(8, 5))
    plt.bar(clases, cantidades)
    plt.xlabel("Clases")
    plt.ylabel("Cantidad de instancias")
    plt.title(f"Distribución de clases - {nombre_dataset}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if guardar_en:
        plt.savefig(guardar_en, dpi=300)
    plt.close()


def obtener_metadata_dataset(nombre_dataset, X_train, y_train, X_test=None, y_test=None):
    clases, conteos = np.unique(y_train, return_counts=True)
    idx_max = np.argmax(conteos)
    
    metadata = {
        "dataset": nombre_dataset,
        "cantidad_train": len(y_train),
        "cantidad_test": len(y_test) if y_test is not None else None,
        "clases": ", ".join(map(str, clases)),
        "clase_mayoritaria": str(clases[idx_max]),
    }

    # Detalle por clase
    for c, cnt in zip(clases, conteos):
        metadata[f"clase_{c}"] = cnt

    # Ejemplo de deficit de cada clase frente a la mayoritaria
    max_cnt = conteos[idx_max]
    for c, cnt in zip(clases, conteos):
        metadata[f"deficit_clase_{c}"] = max_cnt - cnt

    return metadata
