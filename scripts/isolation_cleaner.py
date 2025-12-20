from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationCleaner:
    """
    Limpieza de outliers *por clase* con IsolationForest usando SOLO PERCENTIL.
    - Sin modo global.
    - No usa etiquetas -1/1 (no fit_predict): umbral por percentil sobre decision_function.
    - El percentil se aplica por clase: elimina ~p% con peores scores de cada clase.
    - Opcional: normalizar scores a [-1,1] antes de umbralizar.
    """

    @staticmethod
    def limpiarOutliers(
        X, y,
        idx_original=None, 
        contamination="auto",
        n_estimators=200,
        max_samples="auto",
        random_state=42,
        bootstrap=False,

        # siempre por score
        normalizar_scores: bool = False,

        # *** ÚNICO parámetro de corte ***
        percentil_umbral: float = 5.0,   # ej: 5 => elimina ~5% peor por clase

        devolver_info=False,
        verbose=True,
    ):
        # -------- validaciones --------
        if percentil_umbral is None:
            raise ValueError("percentil_umbral es obligatorio (p. ej. 5.0).")
        p = float(percentil_umbral)
        if not (0.0 <= p <= 100.0):
            raise ValueError("percentil_umbral debe estar en [0, 100].")

        X = np.asarray(X); y = np.asarray(y)
        original_len = len(y)

        if idx_original is None:
            idx_original = np.arange(len(y), dtype=int)
        else:
            idx_original = np.asarray(idx_original, dtype=int)

        def _fit_scores(Xsub):
            iforest = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                random_state=random_state,
                bootstrap=bootstrap,
                n_jobs=1
            )
            iforest.fit(Xsub)
            s = iforest.decision_function(Xsub)  # >0 inlier; <0 outlier
            if normalizar_scores:
                s_min, s_max = np.min(s), np.max(s)
                s = (2.0 * (s - s_min) / (s_max - s_min) - 1.0) if s_max > s_min else np.zeros_like(s)
            return s

        keep_mask = np.zeros_like(y, dtype=bool)
        removed_total = 0
        clases = np.unique(y)
        scores_full = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)
        umbral_usado_por_clase = {}

        for c in clases:
            idx = np.where(y == c)[0]
            # Evitar inestabilidad con clases muy chicas
            if len(idx) < 10:
                keep_mask[idx] = True
                umbral_usado_por_clase[c] = None
                continue

            s_local = _fit_scores(X[idx])

            # Umbral por percentil (cola baja = peor score)
            thr = float(np.percentile(s_local, p))
            keep_local = (s_local >= thr)

            scores_full[idx] = s_local
            umbral_usado_por_clase[c] = thr
            keep_mask[idx] = keep_local
            removed_total += int(np.sum(~keep_local))

        X_clean = X[keep_mask]
        y_clean = y[keep_mask]
        idx_original_clean = idx_original[keep_mask]

        if verbose:
            print(f"🧹 IF (por_clase, percentil={p}%): removidos {removed_total}; quedan {len(y_clean)} de {original_len}.")

        if devolver_info:
            info = {
                "scores": scores_full[keep_mask],
                "idx_keep": idx_original_clean,
                "idx_removed": np.where(~keep_mask)[0],
                "umbral_por_clase": umbral_usado_por_clase,
                "removed_total": removed_total,
                "percentil_umbral": p,
                "normalizar_scores": bool(normalizar_scores),
            }
            return X_clean, y_clean, idx_original_clean, info
        
        return X_clean, y_clean, idx_original_clean
