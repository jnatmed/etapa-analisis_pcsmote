# scripts/gestor_cache.py
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors
from cache import PCSMOTECache
from Utils import Utils


class PCSMOTEGestorCache:
    """
    Gestor intermedio entre PCSMOTE (algoritmo) y PCSMOTECache (persistencia).

    Administra los cómputos costosos reutilizables:
        • Sigmas (auto-escalado LSD): sigma_X y sigma_Xmin
        • Vecindarios globales y locales (por LSD)
        • Umbrales LSD (locales por semilla y global)
        • Matriz de distancias LSD hacia k vecinos globales (n_min × k)
        • Densidades por intersección de esferas (vector n_min)

    Estrategia:
      - Intenta cargar resultados desde caché (clave extendida con fingerprint de la minoritaria)
      - Si el caché es válido, reconstruye umbrales y retorna
      - Si no, calcula todo y guarda en caché (formato v2: carpeta <key>/ con .npy + meta.json)

    Requiere un adaptador (típicamente la instancia del algoritmo) con:
        _compute_sigmas(X, k_sigma)
        _dists_lsd_seed(xi, X_ref, sigma_i, sigmas_ref)
        calcularUmbralDensidades(X_min, vecinos_min_local, percentil, k_sigma)
        calcular_densidad_interseccion(X_min, vecinos_local)
    """

    def __init__(self, cache: Optional[PCSMOTECache], k: int, metrica_vecindario: str, percentil_dist: float):
        self.cache = cache
        self.k = int(k)
        self.metrica = str(metrica_vecindario)
        self.percentil_dist = float(percentil_dist)

    # ----------------------- Wrappers (cache si existe, si no Utils) -----------------------
    def _make_key(self, X_ref: np.ndarray, dataset: str, k: int, metric: str, extra: Dict[str, Any]) -> str:
        if self.cache is not None and hasattr(self.cache, "make_key"):
            return self.cache.make_key(X_ref, dataset, k, metric, extra=extra)
        return Utils.make_key_v2(X_ref, dataset, k, metric, extra)

    def _load_npy_if_exists(self, key: str, fname: str):
        if self.cache is not None and hasattr(self.cache, "load_npy_if_exists"):
            return self.cache.load_npy_if_exists(key, fname=fname)
        return Utils.load_npy_if_exists(key, fname)

    def _atomic_save_npy_and_meta(self, key: str, files: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
        if self.cache is not None and hasattr(self.cache, "atomic_save_npy_and_meta"):
            self.cache.atomic_save_npy_and_meta(key, files=files, meta=meta)
            return
        Utils.atomic_save_npy_and_meta(key, files, meta)

    def _now_iso(self) -> str:
        if self.cache is not None and hasattr(self.cache, "now_iso"):
            return self.cache.now_iso()
        return Utils.now_iso()

    def _hash_arr(self, arr) -> Optional[str]:
        if self.cache is not None and hasattr(self.cache, "hash_ndarray"):
            return self.cache.hash_ndarray(arr)
        return Utils.hash_ndarray(arr)

    # compat: referencias antiguas
    @staticmethod
    def _hash_nd_fallback(arr) -> Optional[str]:
        return Utils.hash_ndarray(arr)

    # Distancias LSD empaquetadas (archivo dedicado)
    def _load_lsd_dists(self, X: np.ndarray, dataset: str, k: int, metric: str, extra: Dict[str, Any]):
        if self.cache is not None and hasattr(self.cache, "load_lsd_dists"):
            return self.cache.load_lsd_dists(X=X, dataset=dataset, k=k, metric=metric, extra=extra)
        key = self._make_key(X, dataset, k, f"lsd_dists_{metric}", extra)
        return self._load_npy_if_exists(key, "lsd_dists.npy")

    def _save_lsd_dists(self, X: np.ndarray, dataset: str, k: int, metric: str, lsd_dists: np.ndarray, extra: Dict[str, Any]) -> None:
        if self.cache is not None and hasattr(self.cache, "save_lsd_dists"):
            self.cache.save_lsd_dists(X=X, dataset=dataset, k=k, metric=metric, lsd_dists=lsd_dists, extra=extra)
            return
        key = self._make_key(X, dataset, k, f"lsd_dists_{metric}", extra)
        meta = {
            "version": 2,
            "artifact": "lsd_dists",
            "dataset": dataset,
            "metric": metric,
            "k": int(k),
            "shape": tuple(lsd_dists.shape),
            "dtype": str(lsd_dists.dtype),
            "created_at": self._now_iso(),
            **(extra or {})
        }
        self._atomic_save_npy_and_meta(key, {"lsd_dists.npy": lsd_dists}, meta)

    # -------------------------------------------------------------------------
    # (A) Core: obtener vecindarios y sigmas (con reconstrucción de umbrales)
    # -------------------------------------------------------------------------
    def obtener(
        self, X: np.ndarray, y: np.ndarray, nombre_dataset: str, adaptador
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna:
          vecinos_all_global : (n_min, k)  índices de vecinos (en X) para riesgo/pureza
          vecinos_min_local  : (n_min, k)  índices de vecinos minoritarios (en X_min)
          vecinos_min_global : (n_min, k)  mapeo de vecinos_min_local a índices globales (en X)
          sigma_X            : (n, )       sigma por punto en X
          sigma_Xmin         : (n_min, )   sigma por punto en X_min

        Efectos colaterales en 'adaptador':
          - adaptador._umbral_lsd_by_i : vector (n_min,) con el percentil local
          - adaptador.umbral_distancia : escalar (percentil global)
        """
        X = np.asarray(X)
        y = np.asarray(y)


        setattr(adaptador, "_X_cache_ref", X)

        # ----------------------- minoritaria + fingerprint -----------------------
        idxs_min_global = np.where(y == 1)[0]
        X_min = X[idxs_min_global]
        extra_meta = {
            "n_min": int(len(idxs_min_global)),
            "pos_fp": PCSMOTECache.fp_array(idxs_min_global) if hasattr(PCSMOTECache, "fp_array") else Utils.sha1_text(str(idxs_min_global.tolist())),
        }

        datos_cache: Optional[Dict[str, Any]] = None
        if self.cache is not None and hasattr(self.cache, "load"):
            datos_cache = self.cache.load(
                X, dataset=nombre_dataset, k=self.k, metric=self.metrica, extra=extra_meta
            )

        # -------------------------- usar caché si es válido ----------------------
        if datos_cache is not None:
            vmin = datos_cache["vecinos_min_local"].astype(int, copy=False)
            if vmin.size == 0 or vmin.max(initial=-1) >= len(X_min):
                datos_cache = None  # caché stale → recomputo

        if datos_cache is not None:
            # Restaurar sigmas y vecindarios
            adaptador._sigma_X = datos_cache["sigma_X"].astype(float, copy=False)
            adaptador._sigma_Xmin = datos_cache["sigma_Xmin"].astype(float, copy=False)
            vecinos_all_global = datos_cache["vecinos_all_global"].astype(int, copy=False)
            vecinos_min_local = datos_cache["vecinos_min_local"].astype(int, copy=False)
            vecinos_min_global = idxs_min_global[vecinos_min_local]

            # Reconstrucción de umbrales LSD (locales y global)
            n_min = len(X_min)
            adaptador._umbral_lsd_by_i = np.full(n_min, np.nan, dtype=float)
            todas_lsd = []
            for i in range(n_min):
                idx_vecinos = vecinos_min_local[i]
                if idx_vecinos.size == 0:
                    continue
                dist_lsd = adaptador._dists_lsd_seed(
                    X_min[i],
                    X_min[idx_vecinos],
                    float(adaptador._sigma_Xmin[i]),
                    adaptador._sigma_Xmin[idx_vecinos],
                )
                if dist_lsd.size:
                    todas_lsd.append(dist_lsd)
                    adaptador._umbral_lsd_by_i[i] = float(np.percentile(dist_lsd, self.percentil_dist))

            adaptador.umbral_distancia = (
                float(np.percentile(np.concatenate(todas_lsd), self.percentil_dist))
                if len(todas_lsd) else 0.0
            )

            return (
                vecinos_all_global,
                vecinos_min_local,
                vecinos_min_global,
                adaptador._sigma_X,
                adaptador._sigma_Xmin,
            )

        # ------------------------------ recomputo -------------------------------
        n_min = len(X_min)
        if n_min == 0:
            return (
                np.empty((0, self.k), dtype=int),
                np.empty((0, self.k), dtype=int),
                np.empty((0, self.k), dtype=int),
                np.ones(len(X), dtype=float),
                np.empty(0, dtype=float),
            )

        k_loc = max(1, min(self.k, n_min - 1))

        # Vecindario preliminar minoritario (índices en X_min)
        nn_min_pre = NearestNeighbors(n_neighbors=k_loc + 1).fit(X_min)
        vecinos_min_local_pre = nn_min_pre.kneighbors(X_min, return_distance=False)[:, 1:]

        # Sigmas (LSD)
        k_sigma_all = max(1, min(self.k, len(X) - 1))
        adaptador._sigma_X    = adaptador._compute_sigmas(X,     k_sigma=k_sigma_all)
        adaptador._sigma_Xmin = adaptador._compute_sigmas(X_min, k_sigma=k_loc)

        # Umbrales locales y global
        adaptador.calcularUmbralDensidades(
            X_min=X_min,
            vecinos_min_local=vecinos_min_local_pre,
            percentil=self.percentil_dist,
            k_sigma=k_loc,
        )

        # Vecindarios LSD (globales en X y locales en X_min), excluyendo self
        vecinos_all_global = np.empty((n_min, self.k), dtype=int)
        vecinos_min_local  = np.empty((n_min, self.k), dtype=int)

        for i, xi in enumerate(X_min):
            d_all = adaptador._dists_lsd_seed(xi, X,     adaptador._sigma_Xmin[i], adaptador._sigma_X)
            d_min = adaptador._dists_lsd_seed(xi, X_min, adaptador._sigma_Xmin[i], adaptador._sigma_Xmin)
            d_all[idxs_min_global[i]] = np.inf
            d_min[i] = np.inf

            k_all = max(1, min(self.k, len(d_all) - 1))
            k_m   = max(1, min(self.k, len(d_min) - 1))

            idx_all = np.argpartition(d_all, k_all - 1)[:k_all]
            idx_min = np.argpartition(d_min, k_m - 1)[:k_m]

            if k_all < self.k:
                pad = np.full(self.k - k_all, idx_all[0], dtype=int)
                vecinos_all_global[i] = np.concatenate([idx_all, pad])[: self.k]
            else:
                vecinos_all_global[i] = idx_all[: self.k]

            if k_m < self.k:
                pad = np.full(self.k - k_m, idx_min[0], dtype=int)
                vecinos_min_local[i] = np.concatenate([idx_min, pad])[: self.k]
            else:
                vecinos_min_local[i] = idx_min[: self.k]

        vecinos_min_global = idxs_min_global[vecinos_min_local.astype(int)]

        # Persistir paquete (si el cache lo soporta)
        if self.cache is not None and hasattr(self.cache, "save"):
            self.cache.save(
                X,
                dataset=nombre_dataset,
                k=self.k,
                metric=self.metrica,
                sigma_X=adaptador._sigma_X,
                sigma_Xmin=adaptador._sigma_Xmin,
                vecinos_all_global=vecinos_all_global,
                vecinos_min_local=vecinos_min_local,
                extra_meta=extra_meta,
            )

        return (
            vecinos_all_global,
            vecinos_min_local,
            vecinos_min_global,
            adaptador._sigma_X,
            adaptador._sigma_Xmin,
        )

    # -------------------------------------------------------------------------
    # (B) LSD dists (n_min × k) hacia vecinos globales, con caché v2
    # -------------------------------------------------------------------------
    def get_or_compute_lsd_dists(
        self,
        X: np.ndarray,
        X_min: np.ndarray,
        sigma_X: np.ndarray,
        sigma_Xmin: np.ndarray,
        k: int,
        vecinos_all_global: np.ndarray,
        adaptador,
        nombre_dataset: str = "unknown",
        extra_meta: Optional[dict] = None,
    ) -> np.ndarray:
        extra_meta = extra_meta or {}

        # 1) intentar cargar
        arr = self._load_lsd_dists(
            X=X,
            dataset=nombre_dataset,
            k=self.k,
            metric=self.metrica,
            extra=extra_meta,
        )
        if arr is not None:
            return arr

        # 2) calcular
        X = np.asarray(X)
        X_min = np.asarray(X_min)
        sigma_X = np.asarray(sigma_X)
        sigma_Xmin = np.asarray(sigma_Xmin)
        vecinos_all_global = np.asarray(vecinos_all_global, dtype=int)

        n_min = len(X_min)
        out = np.empty((n_min, k), dtype=np.float32)
        for i, idxs in enumerate(vecinos_all_global):
            xi = X_min[i]
            sigma_i = float(sigma_Xmin[i])
            sig_ref = sigma_X[idxs]
            d = adaptador._dists_lsd_seed(xi, X[idxs], sigma_i, sig_ref)
            out[i, : len(d)] = d

        # 3) guardar
        self._save_lsd_dists(
            X=X,
            dataset=nombre_dataset,
            k=self.k,
            metric=self.metrica,
            lsd_dists=out,
            extra=extra_meta,
        )
        return out

    # -------------------------------------------------------------------------
    # (C) Densidades (n_min,) por intersección de esferas, con v2 + meta.json
    # -------------------------------------------------------------------------
    def get_or_compute_densidades_v2(
        self,
        *,
        X_min: np.ndarray,
        vecinos_local: np.ndarray,
        sigma_Xmin: Optional[np.ndarray],
        umbrales_lsd_by_i: np.ndarray,
        k: int,
        nombre_dataset: str,
        metric_tag: str,
        adaptador,
        extra_meta: Optional[dict] = None,
    ) -> np.ndarray:
        import os, json
        from pathlib import Path
        import numpy as np

        X_min = np.asarray(X_min)
        vecinos_local = np.asarray(vecinos_local, dtype=int)
        umbrales_lsd_by_i = np.asarray(umbrales_lsd_by_i, dtype=float)
        extra_meta = extra_meta or {}

        if self.cache is None:
            return adaptador.calcular_densidad_interseccion(X_min, vecinos_local)

        # Usar la misma clave/carpeta que el resto de artefactos (sigmas/vecinos)
        X_global = getattr(adaptador, "_X_cache_ref", None)
        if X_global is None:
            X_global = X_min
            
        key = self.cache.make_key(
            X=X_global,
            dataset=nombre_dataset,
            k=int(k),
            metric="lsd",
            extra=extra_meta,
        )

        dirpath: Path = self.cache._dir_for(key)
        dirpath.mkdir(parents=True, exist_ok=True)
        npy_path = dirpath / "densidades.npy"
        meta_path = dirpath / "meta.json"

        # Si ya existe, leer y devolver sin tocar meta
        if npy_path.exists():
            try:
                return np.load(npy_path, allow_pickle=False)
            except Exception:
                pass  # si falla, se recalcula

        # Calcular densidades
        densidades = adaptador.calcular_densidad_interseccion(X_min, vecinos_local)

        # --- MERGE DE META SIN PISAR CAMPOS GLOBALES ---
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        # Garantizar encabezado global coherente
        meta.setdefault("dataset", nombre_dataset or "unknown")
        meta.setdefault("shape", tuple(np.asarray(X_global).shape))          # (n, d) de X
        meta.setdefault("k", int(k))
        meta.setdefault("metric_vecindario", "lsd")
        # usar la misma versión que maneja PCSMOTECache (NO pisar si ya está)
        meta.setdefault("version", int(getattr(self.cache, "version", 1)))
        meta.setdefault("storage_format", "v2/npy")

        # Sección específica del artefacto 'densidades' (ANIDADA)
        meta["densidades"] = {
            "shape": list(np.asarray(densidades).shape),
            "dtype": str(np.asarray(densidades).dtype),
            "metric_tag": str(metric_tag),
            "storage": "v2/npy",
        }
        # conservar extra_meta útil
        for kx, vx in (extra_meta or {}).items():
            meta.setdefault(kx, vx)

        # Escritura atómica
        tmp_npy  = dirpath / "densidades.tmp"
        with open(tmp_npy, "wb") as f:
            np.save(f, np.asarray(densidades), allow_pickle=False)
        os.replace(tmp_npy, npy_path)

        tmp_meta = meta_path.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_meta, meta_path)

        return densidades
