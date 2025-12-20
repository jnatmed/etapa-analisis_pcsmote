# ======================= meta_pcsmote.py (o mismo archivo) =======================

from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import numpy as np
import pandas as pd

class GeneradorMetaPCSMOTE:
    """
    Encapsula el cálculo y la construcción de X_res_meta / y_res_meta.
    No modifica el sampler; sólo recibe los datos y parámetros necesarios.
    """

    def __init__(self):
        self._entrada_fue_dataframe = False
        self._nombres_columnas_entrada = None

    # ---------- Gestión de formato ----------

    def detectar_formato_entrada(self, X):
        """Recuerda si la entrada original fue DataFrame para preservar nombres."""
        self._entrada_fue_dataframe = isinstance(X, pd.DataFrame)
        self._nombres_columnas_entrada = list(X.columns) if self._entrada_fue_dataframe else None

    def reconstruir_dataframe_meta(self, X_ext_np):
        """Devuelve DataFrame con nombres + columnas meta si la entrada fue DataFrame."""
        if self._entrada_fue_dataframe and self._nombres_columnas_entrada:
            cols = self._nombres_columnas_entrada + ["meta_riesgo", "meta_densidad", "meta_pureza"]
            return pd.DataFrame(X_ext_np, columns=cols)
        return X_ext_np

    # ---------- Cálculos de vecindarios y métricas ----------

    def _vecindario_global(self, X, K, metric):
        nn = NearestNeighbors(n_neighbors=K + 1, metric=metric).fit(X)
        d, i = nn.kneighbors(X, return_distance=True)
        return d[:, 1:], i[:, 1:]  # excluyo self

    def _riesgo_general(self, vecinos_idx, y):
        """1 - proporción de vecinos de la misma clase (binario/multiclase)."""
        n = len(y); K = vecinos_idx.shape[1]
        riesgo = np.zeros(n, dtype=float)
        for i in range(n):
            v = vecinos_idx[i]
            prop_misma = np.mean(y[v] == y[i]) if K > 0 else 0.0
            riesgo[i] = 1.0 - float(prop_misma)
        return riesgo

    def _pureza_general(self, vecinos_idx, y, criterio_pureza):
        """
        'entropia': entropía base 2 de clases en el vecindario.
        'proporcion': proporción de vecinos de la MISMA clase.
        """
        n = len(y); pureza = np.zeros(n, dtype=float)
        if criterio_pureza == "entropia":
            for i in range(n):
                clases, counts = np.unique(y[vecinos_idx[i]], return_counts=True)
                p = counts / counts.sum()
                pureza[i] = float(entropy(p, base=2))
        elif criterio_pureza == "proporcion":
            for i in range(n):
                v = vecinos_idx[i]
                pureza[i] = float(np.mean(y[v] == y[i])) if len(v) else 0.0
        else:
            pureza[:] = 0.0
        return pureza

    def _densidad_en_su_clase(self, X, y, K, metric, fn_densidad_interseccion):
        """
        Densidad por intersección dentro de la propia clase de cada muestra.
        Si una clase tiene < K+1, densidad = 0.0.
        'fn_densidad_interseccion' es el método existente del sampler.
        """
        X = np.asarray(X); y = np.asarray(y)
        n = len(y); densidad = np.zeros(n, dtype=float)

        for clase in np.unique(y):
            idx = np.where(y == clase)[0]
            if len(idx) < (K + 1):
                densidad[idx] = 0.0
                continue

            Xc = X[idx]
            nnc = NearestNeighbors(n_neighbors=K + 1, metric=metric).fit(Xc)
            d_c, i_c = nnc.kneighbors(Xc, return_distance=True)
            d_c = d_c[:, 1:]           # (n_c, K)
            i_c_local = i_c[:, 1:]     # (n_c, K)

            dens_c = fn_densidad_interseccion(
                X_min=Xc,
                vecinos_local=i_c_local,
                dists_min_local=d_c
            )
            densidad[idx] = dens_c

        return densidad

    # ---------- Orquestador principal ----------

    def construir_X_y_con_meta(self, *,
                               X_res, y_res, K, metric, criterio_pureza,
                               fn_densidad_interseccion):
        """
        Calcula meta_riesgo, meta_densidad, meta_pureza sobre (X_res, y_res) y
        devuelve (X_res_meta, y_res_meta). No altera los originales.
        """
        if X_res is None or y_res is None or len(y_res) == 0:
            return None, None

        X_np = np.asarray(X_res); y_np = np.asarray(y_res)

        # Vecindario global para riesgo/pureza
        _, vecinos_idx = self._vecindario_global(X_np, K, metric)

        meta_riesgo   = self._riesgo_general(vecinos_idx, y_np)
        meta_densidad = self._densidad_en_su_clase(X_np, y_np, K, metric, fn_densidad_interseccion)
        meta_pureza   = self._pureza_general(vecinos_idx, y_np, criterio_pureza)

        X_ext_np = np.hstack([
            X_np,
            meta_riesgo.reshape(-1, 1),
            meta_densidad.reshape(-1, 1),
            meta_pureza.reshape(-1, 1)
        ])

        X_ext = self.reconstruir_dataframe_meta(X_ext_np)
        return X_ext, y_np.copy()
