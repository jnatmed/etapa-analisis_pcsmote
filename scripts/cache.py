# cache.py
from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from Utils import Utils


class PCSMOTECache:
    """
    Capa de persistencia en disco para reusar cómputos entre corridas
    del mismo dataset (vecindarios y sigmas, y opcionalmente 'candidatos_*').

    Guarda/carga:
      • sigma_X, sigma_Xmin
      • vecinos_all_global, vecinos_min_local
      • (opcional) candidatos_all, candidatos_min

    Clave: (dataset, shape, k, metric, fingerprint y 'extra' opcional).
    Formatos soportados:
      • v2 (preferido): Carpeta <key>/ con .npy + meta.json (permite mmap_mode='r')
      • v1 (legacy):    Archivo .npz comprimido + meta JSON embebido
    """

    def __init__(self,
                 cache_dir: str = os.path.join(os.path.dirname(__file__), "cache"),
                 read: bool = True,
                 write: bool = True,
                 version: int = 1,
                 prefer_npy: bool = True,
                 cleanup_legacy_npz: bool = False):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.read = bool(read)
        self.write = bool(write)
        self.version = int(version)
        self.prefer_npy = bool(prefer_npy)
        self.cleanup_legacy_npz = bool(cleanup_legacy_npz)


    # ---------- helpers reutilizables desde Utils ----------
    @staticmethod
    def now_iso():
        return Utils.now_iso()

    @staticmethod
    def hash_ndarray(arr) -> str | None:
        return Utils.hash_ndarray(arr)

    def load_npy_if_exists(self, key: str, fname: str):
        # key puede ser ruta absoluta o hash; Utils resuelve ambas
        return Utils.load_npy_if_exists(key, fname)

    def atomic_save_npy_and_meta(self, key: str, files: dict[str, np.ndarray], meta: dict):
        Utils.atomic_save_npy_and_meta(key, files, meta)

    # ---------- helpers de clave y path ----------

    @staticmethod
    def fp_array(arr: np.ndarray, max_elems: int = 4096) -> str:
        """Fingerprint SHA1 abreviado de un array 1D (para distinguir particiones)."""
        arr = np.asarray(arr, dtype=np.int64).ravel()
        n = min(max_elems, arr.size)
        return hashlib.sha1(arr[:n].tobytes()).hexdigest()[:12]

    def _fingerprint(self, X: np.ndarray, max_elems: int = 8192) -> str:
        """SHA1 de un preview de X (hasta 8k elementos) para evitar hashear todo."""
        X = np.asarray(X)
        flat = X.ravel()
        n = min(max_elems, flat.size)
        return hashlib.sha1(flat[:n].tobytes()).hexdigest()[:12]


    # ---------- clave corta y válida (carpeta hasheada) ----------
    """
    Este metodo genera una clave legible para facilitar debugging y evitar colisiones
    accidentales. La clave incluye:
      - dataset, shape, k, metric, fingerprint y 'extra'
    """
    def make_key(self,
                X: np.ndarray,
                dataset: str,
                k: int,
                metric: str,
                extra: Optional[Dict[str, Any]] = None) -> str:
        # Genera una clave legible para la caché
        shape = f"{X.shape[0]}x{X.shape[1]}"
        # fingerprint es un metodo que genera un hash corto de X
        # como bien dice una huella digital
        fp = self._fingerprint(X)
        parts = [dataset or "unknown", shape, f"k{k}", str(metric), f"fp{fp}"]
        if extra:
            for kx in sorted(extra.keys()):
                parts.append(f"{kx}={extra[kx]}")
        return "__".join(parts)


    def path_for(self, key: str) -> Path:
        """Ruta legacy v1 (.npz) – igual que antes."""
        return self.cache_dir / f"{key}.npz"

    def _dir_for(self, key: str) -> Path:
        """Ruta v2 (carpeta con .npy + meta.json) – igual que antes."""
        return self.cache_dir / key

    # ---------- utilidades internas ----------

    def _rm_tree(self, root: Path) -> None:
        """Elimina recursivamente un directorio (silencioso)."""
        if not root.exists():
            return
        for p in root.glob("**/*"):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            for p in sorted(root.glob("**/*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            root.rmdir()
        except Exception:
            pass

    # ---------- API pública ----------

    def load(self,
             X: np.ndarray,
             dataset: str,
             k: int,
             metric: str,
             extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Intenta cargar desde caché:
          1) v2 (carpeta con .npy + meta.json) → usa mmap_mode='r'
          2) v1 (.npz comprimido)              → fallback
        Valida shape, k, metric, version y los campos de 'extra' relevantes.
        """
        if not self.read:
            return None

        key = self.make_key(X, dataset, k, metric, extra)
        dirpath = self._dir_for(key)
        npzpath = self.path_for(key)

        # ---- v2: carpeta + .npy (mmap real) ----
        if dirpath.exists() and dirpath.is_dir():
            try:
                meta_path = dirpath / "meta.json"
                if not meta_path.exists():
                    return None
                meta = json.loads(meta_path.read_text(encoding="utf-8"))

                # Validaciones mínimas
                if tuple(meta.get("shape", ())) != tuple(X.shape):
                    return None
                if int(meta.get("k", -1)) != int(k):
                    return None
                if str(meta.get("metric_vecindario", "")) != str(metric):
                    return None
                if int(meta.get("version", -1)) != int(self.version):
                    return None

                # Validación de ‘extra’ si se proveyó
                if extra is not None:
                    for field in ("n_min", "pos_fp"):
                        if field in extra and meta.get(field) != extra[field]:
                            return None

                # formato de almacenamiento (por si queremos debuguear)
                if "storage_format" not in meta:
                    meta["storage_format"] = "v2/npy"

                def _load_npy(name: str):
                    f = dirpath / f"{name}.npy"
                    if not f.exists():
                        raise FileNotFoundError(f)
                    return np.load(f, mmap_mode='r', allow_pickle=False)

                out = {
                    "sigma_X": _load_npy("sigma_X"),
                    "sigma_Xmin": _load_npy("sigma_Xmin"),
                    "vecinos_all_global": _load_npy("vecinos_all_global"),
                    "vecinos_min_local": _load_npy("vecinos_min_local"),
                    "meta": meta,
                }
                # opcionales
                opt_all = dirpath / "candidatos_all.npy"
                opt_min = dirpath / "candidatos_min.npy"
                if opt_all.exists():
                    out["candidatos_all"] = np.load(opt_all, mmap_mode='r', allow_pickle=False)
                if opt_min.exists():
                    out["candidatos_min"] = np.load(opt_min, mmap_mode='r', allow_pickle=False)

                return out
            except Exception:
                return None

        # ---- v1: .npz (legacy, sin mmap efectivo) ----
        if not npzpath.exists():
            return None
        try:
            with np.load(npzpath, allow_pickle=False) as data:
                meta = json.loads(data["meta"].tobytes().decode("utf-8"))

                if tuple(meta.get("shape", ())) != tuple(X.shape):
                    return None
                if int(meta.get("k", -1)) != int(k):
                    return None
                if str(meta.get("metric_vecindario", "")) != str(metric):
                    return None
                if int(meta.get("version", -1)) != int(self.version):
                    return None

                    
                # Validación de ‘extra’ si se proveyó (robustez adicional)
                if extra is not None:
                    for field in ("n_min", "pos_fp"):
                        if field in extra and meta.get(field) != extra[field]:
                            return None

                if "storage_format" not in meta:
                    meta["storage_format"] = "v1/npz"

                out = {
                    "sigma_X": data["sigma_X"],
                    "sigma_Xmin": data["sigma_Xmin"],
                    "vecinos_all_global": data["vecinos_all_global"],
                    "vecinos_min_local": data["vecinos_min_local"],
                    "meta": meta,
                }

                # --- AUTO-MIGRAR A v2 SI SE PREFIERE NPY ---
                if self.prefer_npy:
                    try:
                        dirpath.mkdir(parents=True, exist_ok=True)

                        # asegurar el flag para trazas
                        meta["storage_format"] = "v2/npy"
                        # guardar .npy
                        np.save(dirpath / "sigma_X.npy",        out["sigma_X"],        allow_pickle=False)
                        np.save(dirpath / "sigma_Xmin.npy",     out["sigma_Xmin"],     allow_pickle=False)
                        np.save(dirpath / "vecinos_all_global.npy", out["vecinos_all_global"], allow_pickle=False)
                        np.save(dirpath / "vecinos_min_local.npy",  out["vecinos_min_local"],  allow_pickle=False)
                        # opcionales si existen en out
                        if "candidatos_all" in out:
                            np.save(dirpath / "candidatos_all.npy", out["candidatos_all"], allow_pickle=False)
                        if "candidatos_min" in out:
                            np.save(dirpath / "candidatos_min.npy", out["candidatos_min"], allow_pickle=False)

                        (dirpath / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

                        # limpieza legacy si se pidió
                        if self.cleanup_legacy_npz and npzpath.exists():
                            try:
                                npzpath.unlink(missing_ok=True)
                            except Exception:
                                pass
                    except Exception:
                        # si falla la migración, no rompemos la carga
                        pass


                if "candidatos_all" in data.files:
                    out["candidatos_all"] = data["candidatos_all"]
                if "candidatos_min" in data.files:
                    out["candidatos_min"] = data["candidatos_min"]
                return out
        except Exception:
            return None

    def save(self,
             X: np.ndarray,
             dataset: str,
             k: int,
             metric: str,
             sigma_X: np.ndarray,
             sigma_Xmin: np.ndarray,
             vecinos_all_global: np.ndarray,
             vecinos_min_local: np.ndarray,
             candidatos_all: Optional[np.ndarray] = None,
             candidatos_min: Optional[np.ndarray] = None,
             extra_meta: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Guarda en caché:
          • v2 (por defecto): carpeta <key>/ con .npy y meta.json (write atómico)
          • v1 (legacy): .npz comprimido
        """
        if not self.write:
            return None

        key = self.make_key(X, dataset, k, metric, extra=extra_meta)
        dirpath = self._dir_for(key)
        npzpath = self.path_for(key)

        meta = {
            "dataset": dataset or "unknown",
            "shape": X.shape,
            "k": int(k),
            "metric_vecindario": str(metric),
            "version": int(self.version),
        }
        if extra_meta:
            meta.update(extra_meta)

        if self.prefer_npy:
            # ---- v2: guardar en carpeta con .npy + meta.json (ATÓMICO) ----
            tmpdir = dirpath.with_suffix(".tmp")
            try:
                # limpiar restos de intentos previos
                if tmpdir.exists():
                    self._rm_tree(tmpdir)
                tmpdir.mkdir(parents=True, exist_ok=True)

                def _save_npy(name: str, arr: np.ndarray):
                    np.save(tmpdir / f"{name}.npy", np.asarray(arr), allow_pickle=False)

                _save_npy("sigma_X", sigma_X)
                _save_npy("sigma_Xmin", sigma_Xmin)
                _save_npy("vecinos_all_global", np.asarray(vecinos_all_global, dtype=np.int64))
                _save_npy("vecinos_min_local", np.asarray(vecinos_min_local, dtype=np.int64))
                if candidatos_all is not None:
                    _save_npy("candidatos_all", np.asarray(candidatos_all, dtype=np.int64))
                if candidatos_min is not None:
                    _save_npy("candidatos_min", np.asarray(candidatos_min, dtype=np.int64))

                meta["storage_format"] = "v2/npy"
                (tmpdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

                # movimiento atómico: reemplaza la carpeta anterior
                if dirpath.exists():
                    self._rm_tree(dirpath)
                tmpdir.rename(dirpath)

                # limpieza opcional del .npz legacy
                if self.cleanup_legacy_npz and npzpath.exists():
                    try:
                        npzpath.unlink(missing_ok=True)
                    except Exception:
                        pass

                return dirpath
            except Exception:
                # si falla v2, no dejamos basura
                try:
                    self._rm_tree(tmpdir)
                except Exception:
                    pass
                return None

        # ---- v1: guardar .npz comprimido (retrocompatibilidad) ----
        meta["storage_format"] = "v1/npz"
        try:
            arrays = dict(
                sigma_X=np.asarray(sigma_X),
                sigma_Xmin=np.asarray(sigma_Xmin),
                vecinos_all_global=np.asarray(vecinos_all_global, dtype=np.int64),
                vecinos_min_local=np.asarray(vecinos_min_local, dtype=np.int64),
                meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            )
            if candidatos_all is not None:
                arrays["candidatos_all"] = np.asarray(candidatos_all, dtype=np.int64)
            if candidatos_min is not None:
                arrays["candidatos_min"] = np.asarray(candidatos_min, dtype=np.int64)

            np.savez_compressed(npzpath, **arrays)
            return npzpath
        except Exception:
            return None

    # ---------- utilidades ----------

    def clear_key(self,
                  X: np.ndarray,
                  dataset: str,
                  k: int,
                  metric: str,
                  extra: Optional[Dict[str, Any]] = None) -> bool:
        """Elimina la entrada de caché (v2 y v1) correspondiente a la clave."""
        key = self.make_key(X, dataset, k, metric, extra)
        dirpath = self._dir_for(key)
        npzpath = self.path_for(key)
        removed = False

        if dirpath.exists():
            self._rm_tree(dirpath)
            removed = True
        if npzpath.exists():
            try:
                npzpath.unlink(missing_ok=True)
                removed = True
            except Exception:
                pass
        return removed

    def clear_all(self) -> int:
        """Elimina todas las entradas de caché (carpetas v2 y archivos .npz)."""
        cnt = 0
        # v1
        for p in self.cache_dir.glob("*.npz"):
            try:
                p.unlink(missing_ok=True)
                cnt += 1
            except Exception:
                pass
        # v2
        for d in self.cache_dir.iterdir():
            if d.is_dir():
                try:
                    self._rm_tree(d)
                    cnt += 1
                except Exception:
                    pass
        return cnt

    # --- Artefacto: lsd_dists (v2 .npy + meta.json) ------------------------------

    def load_lsd_dists(self,
                    X: np.ndarray,
                    dataset: str,
                    k: int,
                    metric: str,
                    extra: Optional[Dict[str, Any]] = None):
        """
        Carga la matriz (n_min, k) de distancias LSD precomputadas en formato v2
        (<key>/lsd_dists.npy + meta.json). Devuelve np.memmap (mmap_mode='r')
        o None si no existe o no valida.
        """
        if not self.read:
            return None

        key = self.make_key(X, dataset, k, metric, extra)
        dirpath = self._dir_for(key)
        meta_path = dirpath / "meta.json"
        lsd_path = dirpath / "lsd_dists.npy"

        # Verificación de existencia básica
        if not (dirpath.is_dir() and meta_path.exists() and lsd_path.exists()):
            return None

        try:
            # Leer meta y validar campos mínimos (alineado con load())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            if tuple(meta.get("shape", ())) != tuple(X.shape):
                return None
            if int(meta.get("k", -1)) != int(k):
                return None
            if str(meta.get("metric_vecindario", "")) != str(metric):
                return None
            if int(meta.get("version", -1)) != int(self.version):
                return None

            # Validación de ‘extra’ si se proveyó
            if extra is not None:
                for field in ("n_min", "pos_fp"):
                    if field in extra and meta.get(field) != extra[field]:
                        return None

            # Carga con mmap real (solo lectura)
            return np.load(lsd_path, mmap_mode="r", allow_pickle=False)
        except Exception:
            # En cualquier problema de lectura/validación, retornar None
            return None



    def save_lsd_dists(self,
                    X: np.ndarray,
                    dataset: str,
                    k: int,
                    metric: str,
                    lsd_dists: np.ndarray,
                    extra: Optional[Dict[str, Any]] = None):
        """
        Guarda <key>/lsd_dists.npy (v2) de forma atómica en Windows/Linux.
        Escribe a un .tmp (sin .npy) usando file object y luego os.replace.
        Actualiza/mergea meta.json.
        """
        if not self.write:
            return None

        key = self.make_key(X, dataset, k, metric, extra)
        dirpath = self._dir_for(key)
        dirpath.mkdir(parents=True, exist_ok=True)

        # Limpieza de temporales huérfanos previos, si existieran
        for p in dirpath.glob("lsd_dists*.tmp"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        meta_path = dirpath / "meta.json"
        final_npy = dirpath / "lsd_dists.npy"
        tmp_npy   = dirpath / "lsd_dists.tmp"      # ¡sin .npy!
        tmp_meta  = dirpath / "meta.json.tmp"

        arr = np.asarray(lsd_dists)

        # --- escritura del NPY en archivo temporal (file object) ---
        # IMPORTANTÍSIMO: abrir como file object evita que np.save agregue .npy al nombre
        with open(tmp_npy, "wb") as f:
            np.save(f, arr, allow_pickle=False)

        # En Windows hay que asegurarse de que el archivo está cerrado antes del replace
        os.replace(tmp_npy, final_npy)  # atómico dentro del mismo directorio

        # --- meta.json: leer/mergear/escribir atómicamente ---
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        meta.setdefault("dataset", dataset or "unknown")
        meta.setdefault("shape", tuple(X.shape))
        meta.setdefault("k", int(k))
        meta.setdefault("metric_vecindario", str(metric))
        meta.setdefault("version", int(self.version))
        meta.setdefault("storage_format", "v2/npy")
        if extra:
            meta.update(extra)

        # sección específica del artefacto lsd_dists
        meta["lsd_dists"] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "storage": "v2/npy",
        }

        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_meta, meta_path)

        return final_npy
