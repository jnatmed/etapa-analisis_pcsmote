# limpiador.py

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

class LimpiadorOutliers:
    """
    Clase de limpieza quirúrgica basada en configuración.
    - Nivel 1 (rango_fisico): elimina solo valores imposibles/ilógicos. Pensado para correr ANTES de escalar.
    - Nivel 2 (iqr_por_clase): calcula outliers por clase usando IQR y solo marca (no elimina).
    - Nivel 3: placeholder (compatibilidad con config).
    """

    def __init__(self, activar_global: bool = True) -> None:
        # Bandera global: si es False, no se ejecuta ningún nivel (la config por dataset no se toca).
        self.activar_global = bool(activar_global)

    def configurar_bandera(self, activar: bool) -> None:
        # Permite alternar en runtime sin re-instanciar.
        self.activar_global = bool(activar)

    @staticmethod
    def _construir_dataframe_desde_matriz_y_columnas(matriz_features: np.ndarray,
                                                     vector_target: np.ndarray,
                                                     nombres_columnas: List[str]) -> pd.DataFrame:
        df = pd.DataFrame(matriz_features, columns=nombres_columnas)
        df["__target__"] = vector_target
        return df

    @staticmethod
    def _extraer_matriz_y_target(df: pd.DataFrame,
                                 nombres_columnas: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = df[nombres_columnas].to_numpy()
        y = df["__target__"].to_numpy()
        return X, y

    # ---------- regla de habilitación ----------
    def _esta_habilitada(self, bloque_config_limpieza: Dict[str, Any]) -> bool:
        if self.activar_global is not True:
            return False
        if bloque_config_limpieza is None:
            return False
        return bool(bloque_config_limpieza.get("activar", False))

    # ------------------ NIVEL 1: RANGO FÍSICO (ELIMINA) ------------------
    @staticmethod
    def aplicar_nivel_1_rango_fisico(X: np.ndarray,
                                     y: np.ndarray,
                                     nombres_columnas: List[str],
                                     criterios_rango: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        df = LimpiadorOutliers._construir_dataframe_desde_matriz_y_columnas(X, y, nombres_columnas)

        mascara_global_conservar = np.ones(len(df), dtype=bool)
        rangos_aplicados = {}

        indice_variable = 0
        while indice_variable < len(nombres_columnas):
            nombre_variable = nombres_columnas[indice_variable]
            if nombre_variable in criterios_rango:
                limites = criterios_rango[nombre_variable]
                hay_minimo = ("min" in limites)
                hay_maximo = ("max" in limites)

                rangos_aplicados[nombre_variable] = {
                    "min": limites.get("min", None),
                    "max": limites.get("max", None),
                }

                mascara_variable_conservar = np.ones(len(df), dtype=bool)
                indice_fila = 0
                while indice_fila < len(df):
                    valor_actual = float(df.loc[df.index[indice_fila], nombre_variable])
                    cumple_min = True
                    cumple_max = True

                    if hay_minimo:
                        if valor_actual < float(limites["min"]):
                            cumple_min = False
                    if hay_maximo:
                        if valor_actual > float(limites["max"]):
                            cumple_max = False

                    if (cumple_min is True) and (cumple_max is True):
                        mascara_variable_conservar[indice_fila] = True
                    else:
                        mascara_variable_conservar[indice_fila] = False

                    indice_fila += 1

                mascara_global_conservar = mascara_global_conservar & mascara_variable_conservar
            indice_variable += 1

        idx_eliminados = np.where(~mascara_global_conservar)[0].tolist()
        df_limpio = df.loc[mascara_global_conservar].copy()
        X_limpio, y_limpio = LimpiadorOutliers._extraer_matriz_y_target(df_limpio, nombres_columnas)

        info = {
            "nivel": "rango_fisico",
            "idx_eliminados": idx_eliminados,
            "cantidad_eliminados": len(idx_eliminados),
            "rangos_aplicados": rangos_aplicados
        }
        return X_limpio, y_limpio, info

    # ------------------ NIVEL 2: IQR POR CLASE (NO ELIMINA) ------------------
    @staticmethod
    def calcular_iqr_por_clase(X: np.ndarray,
                               y: np.ndarray,
                               nombres_columnas: List[str]) -> Dict[Any, Dict[str, Dict[str, float]]]:
        df = LimpiadorOutliers._construir_dataframe_desde_matriz_y_columnas(X, y, nombres_columnas)
        umbrales = {}

        clases_unicas = np.unique(y)
        indice_clase = 0
        while indice_clase < len(clases_unicas):
            clase_actual = clases_unicas[indice_clase]
            grupo = df[df["__target__"] == clase_actual]
            umbrales_clase = {}

            indice_variable = 0
            while indice_variable < len(nombres_columnas):
                nombre_variable = nombres_columnas[indice_variable]
                serie = grupo[nombre_variable].astype(float)

                q1 = float(np.percentile(serie, 25))
                q3 = float(np.percentile(serie, 75))
                iqr = float(q3 - q1)
                limite_inferior = float(q1 - 1.5 * iqr)
                limite_superior = float(q3 + 1.5 * iqr)

                umbrales_clase[nombre_variable] = {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "limite_inferior": limite_inferior,
                    "limite_superior": limite_superior
                }
                indice_variable += 1

            umbrales[clase_actual] = umbrales_clase
            indice_clase += 1

        return umbrales

    @staticmethod
    def marcar_outliers_iqr_por_clase(X: np.ndarray,
                                      y: np.ndarray,
                                      nombres_columnas: List[str],
                                      umbrales: Dict[Any, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        df = LimpiadorOutliers._construir_dataframe_desde_matriz_y_columnas(X, y, nombres_columnas)

        mask_outlier_por_variable = {}
        indices_outlier_por_variable_y_clase = defaultdict(lambda: defaultdict(list))
        mask_outlier_global = np.zeros(len(df), dtype=bool)

        indice_variable = 0
        while indice_variable < len(nombres_columnas):
            nombre_variable = nombres_columnas[indice_variable]
            mask_variable = np.zeros(len(df), dtype=bool)

            clases_unicas = np.unique(y)
            indice_clase = 0
            while indice_clase < len(clases_unicas):
                clase_actual = clases_unicas[indice_clase]
                um = umbrales[clase_actual][nombre_variable]
                li = um["limite_inferior"]
                ls = um["limite_superior"]

                idx_clase = np.where(y == clase_actual)[0]

                indice_local = 0
                while indice_local < len(idx_clase):
                    idx_global = idx_clase[indice_local]
                    valor = float(df.loc[df.index[idx_global], nombre_variable])
                    es_outlier = (valor < li) or (valor > ls)
                    if es_outlier:
                        mask_variable[idx_global] = True
                        indices_outlier_por_variable_y_clase[clase_actual][nombre_variable].append(int(idx_global))
                    indice_local += 1

                indice_clase += 1

            mask_outlier_por_variable[nombre_variable] = mask_variable
            mask_outlier_global = mask_outlier_global | mask_variable
            indice_variable += 1

        info = {
            "mask_outlier_por_variable": mask_outlier_por_variable,
            "mask_outlier_global": mask_outlier_global,
            "indices_outlier_por_variable_y_clase": indices_outlier_por_variable_y_clase
        }
        return info

    # ------------------ INTERFAZ PRINCIPAL ------------------
    def limpiar_antes_de_escalar_si_corresponde(self, X, y, nombres_columnas, bloque_config_limpieza):
        """
        Nivel 1 de limpieza ANTES de escalar.
        Siempre devuelve: (X_salida, y_salida, info) con 'info' dict no-nulo.
        """
        info = {"nivel_1_aplicado": False, "detalle_nivel_1": {}}

        # 0) Chequeo de habilitación
        if not self._esta_habilitada(bloque_config_limpieza):
            print("ℹ️ [limpieza] Deshabilitada por bandera/global o config nula.")
            return X, y, info

        niveles = (bloque_config_limpieza or {}).get("niveles", {}) or {}
        nivel_1 = niveles.get("nivel_1", {}) or {}

        tipo_n1 = str(nivel_1.get("tipo", "")).strip().lower()
        criterios = nivel_1.get("criterios", {}) or {}

        if tipo_n1 != "rango_fisico" or len(criterios) == 0:
            print("ℹ️ [nivel_1] Sin 'rango_fisico' o sin criterios → no se aplica.")
            return X, y, info

        # 1) Respaldo
        X0, y0 = X, y

        # 2) Ejecutar nivel 1 (intento)
        X1, y1, info_n1 = LimpiadorOutliers.aplicar_nivel_1_rango_fisico(
            X, y, nombres_columnas, criterios
        )

        # 3) Normalización defensiva
        if not isinstance(info_n1, dict):
            info_n1 = {}
        if "cantidad_eliminados" not in info_n1 or not isinstance(info_n1.get("cantidad_eliminados", None), (int, float)):
            info_n1["cantidad_eliminados"] = 0
        if "idx_eliminados" not in info_n1 or not isinstance(info_n1.get("idx_eliminados", None), (list, tuple)):
            info_n1["idx_eliminados"] = []
        if "rangos_aplicados" not in info_n1 or not isinstance(info_n1.get("rangos_aplicados", None), dict):
            info_n1["rangos_aplicados"] = {}

        total = int(len(y0))
        eliminados = int(info_n1.get("cantidad_eliminados", 0))
        ratio = (float(eliminados) / float(total)) if total > 0 else 0.0

        # 4) Siempre loguear qué detectó el intento (aunque luego se aborte)
        print(f"👀 [nivel_1] Detectaría {eliminados} filas fuera de rango "
            f"(ratio={ratio:.4f}; total={total}).")

        # 5) Fail-safe (por defecto 0.0 si no está definido en config)
        fail_safe = nivel_1.get("fail_safe_max_ratio_eliminados")
        try:
            fail_safe = float(fail_safe) if fail_safe is not None else 0.0
        except Exception:
            fail_safe = 0.0

        if ratio > fail_safe:
            # Abortamos → devolvemos dataset intacto PERO logueamos y devolvemos detalle del intento
            print(f"🛑 [nivel_1] ABORTADO por fail-safe: ratio={ratio:.4f} > max={fail_safe:.4f}. "
                f"Intento de eliminar={eliminados}. No se eliminó nada.")
            info["nivel_1_aplicado"] = False
            info["detalle_nivel_1"] = {
                "abortado_por_fail_safe": True,
                "cantidad_eliminados_intento": eliminados,
                "idx_eliminados_intento": list(info_n1.get("idx_eliminados", [])),
                "ratio_eliminados": ratio,
                "rangos_aplicados": info_n1.get("rangos_aplicados", {})
            }
            return X0, y0, info

        # 6) Aplica eliminación (pasa el fail-safe)
        info["nivel_1_aplicado"] = True
        info["detalle_nivel_1"] = {
            "cantidad_eliminados": eliminados,
            "idx_eliminados": list(info_n1.get("idx_eliminados", [])),
            "rangos_aplicados": info_n1.get("rangos_aplicados", {})
        }
        print(f"🧹 [nivel_1] Aplicado (rango_fisico). Eliminados={eliminados} (ratio={ratio:.4f}).")
        return X1, y1, info


    def marcar_outliers_post_split_por_clase(self,
                                             X_train: np.ndarray,
                                             y_train: np.ndarray,
                                             nombres_columnas: List[str],
                                             bloque_config_limpieza: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta Nivel 2 (iqr_por_clase) solo si bandera global y bloque lo permiten.
        Nunca elimina; solo devuelve diagnóstico.
        """
        info = {"nivel_2_aplicado": False, "umbrales": None, "marcacion": None}

        if not self._esta_habilitada(bloque_config_limpieza):
            return info

        niveles = bloque_config_limpieza.get("niveles", {})
        nivel_2 = niveles.get("nivel_2", {})
        activar = bool(nivel_2.get("activar", True))
        tipo_n2 = str(nivel_2.get("tipo", "")).strip().lower()

        if (activar is True) and (tipo_n2 == "iqr_por_clase"):
            umbrales = LimpiadorOutliers.calcular_iqr_por_clase(X_train, y_train, nombres_columnas)
            marcacion = LimpiadorOutliers.marcar_outliers_iqr_por_clase(X_train, y_train, nombres_columnas, umbrales)
            info["nivel_2_aplicado"] = True
            info["umbrales"] = umbrales
            info["marcacion"] = marcacion

        return info

    @staticmethod
    def winsorizar_por_percentiles(X: np.ndarray,
                                   nombres_columnas: List[str],
                                   p_inferior: float,
                                   p_superior: float) -> np.ndarray:
        import numpy as np
        X_salida = X.copy()
        indice_variable = 0
        while indice_variable < len(nombres_columnas):
            valores = X_salida[:, indice_variable].astype(float)
            limite_inferior = np.percentile(valores, p_inferior * 100.0)
            limite_superior = np.percentile(valores, p_superior * 100.0)
            indice_fila = 0
            while indice_fila < X_salida.shape[0]:
                valor_actual = float(X_salida[indice_fila, indice_variable])
                if valor_actual < limite_inferior:
                    X_salida[indice_fila, indice_variable] = limite_inferior
                elif valor_actual > limite_superior:
                    X_salida[indice_fila, indice_variable] = limite_superior
                indice_fila += 1
            indice_variable += 1
        return X_salida

    @staticmethod
    def aplicar_escalado_robusto(X: np.ndarray,
                                 nombres_columnas: List[str]) -> np.ndarray:
        import numpy as np
        X_salida = X.copy()
        indice_variable = 0
        while indice_variable < len(nombres_columnas):
            columna = X_salida[:, indice_variable].astype(float)
            mediana = float(np.median(columna))
            q1 = float(np.percentile(columna, 25.0))
            q3 = float(np.percentile(columna, 75.0))
            iqr = float(q3 - q1) if (q3 - q1) != 0.0 else 1.0  # evitar división por 0
            indice_fila = 0
            while indice_fila < X_salida.shape[0]:
                valor = float(X_salida[indice_fila, indice_variable])
                X_salida[indice_fila, indice_variable] = (valor - mediana) / iqr
                indice_fila += 1
            indice_variable += 1
        return X_salida

    def transformar_despues_de_limpieza(self,
                                        X: np.ndarray,
                                        nombres_columnas: List[str],
                                        bloque_config_transformacion: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = {"winsorizacion_aplicada": False, "escalado_aplicado": None}

        if not isinstance(bloque_config_transformacion, dict):
            return X, info

        # Winsorización (si corresponde)
        wins = bloque_config_transformacion.get("winsorizacion", {})
        if bool(wins.get("aplicar", False)):
            p_inf = float(wins.get("p_inferior", 0.01))
            p_sup = float(wins.get("p_superior", 0.99))
            X = self.winsorizar_por_percentiles(X, nombres_columnas, p_inf, p_sup)
            info["winsorizacion_aplicada"] = True

        # Escalado (robust / standard / minmax futuro)
        esc = bloque_config_transformacion.get("escalado", {})
        if bool(esc.get("aplicar", False)):
            tipo = str(esc.get("tipo", "robust")).strip().lower()
            if tipo == "robust":
                X = self.aplicar_escalado_robusto(X, nombres_columnas)
                info["escalado_aplicado"] = "robust"

        return X, info
