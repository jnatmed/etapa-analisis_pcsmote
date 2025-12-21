import os
from pathlib import Path
import numpy as np
import pandas as pd


class Utils:
    """
    Clase utilitaria para:
    - logging por muestra (semillas)
    - exportar logs a Excel (por muestras)
    """

    def __init__(self):
        # listado de registros por semilla (se resetea en cada fit_resample*_*)
        self.logs_por_muestra = []

    @staticmethod
    def tag_p(criterio_pureza: str) -> str:
        """
        Devuelve un token corto para colocar en el nombre del archivo.
        - entropia    -> entropia
        - proporcion  -> proporcion
        - cualquier otro string -> el string lower(), sin espacios.

        Nada más. No inventa abreviaturas.
        """
        if criterio_pureza is None:
            return "none"

        criterio = str(criterio_pureza).strip().lower()

        if criterio in ("entropia", "proporcion"):
            return criterio

        # fallback genérico
        return criterio.replace(" ", "_")

    @staticmethod
    def safe_token(nombre: str) -> str:
        """
        Limpia un nombre de archivo para que sea seguro en Windows/Linux.

        - Reemplaza espacios por '_'
        - Quita caracteres peligrosos:  : * ? " < > | \
        - Normaliza dobles underscores
        - Mantiene extensión si existe
        """
        if nombre is None:
            return ""

        # Convertir a str por seguridad
        nombre = str(nombre)

        # Reemplazar espacios por underscore
        nombre = nombre.replace(" ", "_")

        # Caracteres ilegales en Windows y Linux
        ilegales = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']

        for c in ilegales:
            nombre = nombre.replace(c, "")

        # Quitar dobles/triples underscores
        while "__" in nombre:
            nombre = nombre.replace("__", "_")

        # Evitar underscore final
        if nombre.endswith("_"):
            nombre = nombre[:-1]

        return nombre


    # ----------------------------------------------------------------------
    # LOG UNITARIO
    # ----------------------------------------------------------------------
    def registrar_log_por_semilla(
        self,
        nombre_configuracion,
        idx_global,
        clase_objetivo,
        clase_real,
        es_semilla_valida,
        k,
        radio_densidad,
        umbral_riesgo,
        umbral_entropia,
        criterio_pureza,
        fraccion_proporcion_min,
        valor_proporcion_min,
        fraccion_densidad,
        valor_densidad,
        fraccion_riesgo,
        valor_riesgo,
        entropia,
        pasa_pureza,
        pasa_densidad,
        pasa_riesgo,
        vecinos_validos_percentil,
        thr_dist_percentil,
        synthetics_from_seed,
        cant_vecinos_en_p,
        cant_min_en_p,
    ):
        """
        Inserta UN registro en logs_por_muestra.

        - fraccion_* son strings tipo "a/k".
        - *_valor son floats en [0,1].
        - umbral_* son valores numéricos de distancia/entropía (no "P80").
        """

        registro = {
            "configuracion": nombre_configuracion,
            "idx_global": int(idx_global),
            "clase_objetivo": clase_objetivo,
            "clase_real_muestra": clase_real,
            "es_semilla_valida": bool(es_semilla_valida),
            "k": int(k),

            "radio_densidad": float(radio_densidad),
            "umbral_riesgo": float(umbral_riesgo),
            "umbral_entropia": (
                float(umbral_entropia) if umbral_entropia is not None else None
            ),
            "criterio_pureza": str(criterio_pureza),

            "proporcion_min": fraccion_proporcion_min,
            "proporcion_min_valor": float(valor_proporcion_min),

            "densidad": fraccion_densidad,
            "densidad_valor": float(valor_densidad),

            "riesgo": fraccion_riesgo,
            "riesgo_valor": float(valor_riesgo),

            "entropia": float(entropia),

            "pasa_pureza": bool(pasa_pureza),
            "pasa_densidad": bool(pasa_densidad),
            "pasa_riesgo": bool(pasa_riesgo),

            "vecinos_validos_por_percentil": int(vecinos_validos_percentil),
            "thr_dist_percentil": float(thr_dist_percentil),
            "synthetics_from_this_seed": int(synthetics_from_seed),
            "cant_vecinos_en_p_elegido": int(cant_vecinos_en_p),
            "cant_min_en_p_elegido": int(cant_min_en_p),
        }

        self.logs_por_muestra.append(registro)

    # ----------------------------------------------------------------------
    # LOG MASIVO (todas las semillas positivas)
    # ----------------------------------------------------------------------
    def loguear_semillas_positivas(
        self,
        nombre_configuracion,
        clase_objetivo,
        y_original,
        y_binaria,
        k,
        indices_positivos,
        indices_vecinos_k,
        distancias_k,
        radio_densidad,
        umbral_densidad,
        radio_riesgo,
        umbral_entropia,
        umbral_riesgo,
        criterio_pureza,
        proporciones_min,
        densidades,
        riesgos,
        entropias,
        mascara_pureza,
        mascara_densidad,
        mascara_riesgo,
        mascara_candidata,
        conteo_sinteticas_por_semilla,
        idx_original_X,
    ):
        """
        Recorre TODAS las semillas positivas y llama a registrar_log_por_semilla
        para cada una. Es el bloque que sacamos de PCSMOTE.
        """
        cantidad_semillas_pos = len(indices_positivos)

        for indice_local_semilla in range(cantidad_semillas_pos):
            indice_global_semilla = int(indices_positivos[indice_local_semilla])
            idx_original_semilla = int(idx_original_X[indice_global_semilla]) # índice en X original

            indices_vecinos_actual = indices_vecinos_k[indice_local_semilla]
            distancias_actual = distancias_k[indice_local_semilla]

            cant_misma = 0
            cant_dentro_densidad = 0
            cant_min_dentro_densidad = 0
            cant_contrarios_riesgo = 0

            for j in range(len(indices_vecinos_actual)):
                idx_v = int(indices_vecinos_actual[j])
                dist_v = float(distancias_actual[j])
                es_misma = (int(y_binaria[idx_v]) == 1)
                es_contraria = not es_misma

                if es_misma:
                    cant_misma += 1
                if dist_v <= float(radio_densidad):
                    cant_dentro_densidad += 1
                    if es_misma:
                        cant_min_dentro_densidad += 1
                if dist_v <= float(radio_riesgo) and es_contraria:
                    cant_contrarios_riesgo += 1

            # Valores numéricos de pureza (según criterio)
            if proporciones_min is not None:
                valor_proporcion_min = float(proporciones_min[indice_local_semilla])
            else:
                valor_proporcion_min = None

            if entropias is not None:
                valor_entropia = float(entropias[indice_local_semilla])
            else:
                valor_entropia = None
            fraccion_proporcion_min = f"{cant_misma}/{k}"
            fraccion_densidad = f"{cant_dentro_densidad}/{k}"
            fraccion_riesgo = f"{cant_contrarios_riesgo}/{k}"

            # valor_proporcion_min = float(proporciones_min[indice_local_semilla])
            valor_densidad = float(densidades[indice_local_semilla])
            valor_riesgo = float(riesgos[indice_local_semilla])
            # valor_entropia = float(entropias[indice_local_semilla])

            pasa_pureza = bool(mascara_pureza[indice_local_semilla])
            pasa_densidad = bool(mascara_densidad[indice_local_semilla])
            pasa_riesgo = bool(mascara_riesgo[indice_local_semilla])
            es_semilla_valida = bool(mascara_candidata[indice_local_semilla])

            self.logs_por_muestra.append(
                {
                    "configuracion": nombre_configuracion,
                    # "idx_global": indice_global_semilla,
                    "idx_original": idx_original_semilla, # índice en X original
                    "clase_objetivo": clase_objetivo,
                    # "clase_real_muestra": y_original[indice_global_semilla],
                    "es_semilla_valida": es_semilla_valida,
                    "k": k,
                    "radio_percentil_distancias": radio_densidad,
                    "umbral_densidad": umbral_densidad,
                    "radio_percentil_riesgos": radio_riesgo,
                    "umbral_entropia": umbral_entropia,
                    "umbral_riesgo": umbral_riesgo,
                    "entropia": valor_entropia,
                    "criterio_pureza": criterio_pureza,
                    # si se midió proporción, fracción y valor; si no, quedan None
                    "proporcion_min": fraccion_proporcion_min if proporciones_min is not None else None,
                    "proporcion_min_valor": valor_proporcion_min,
                    # densidad y riesgo siempre
                    "densidad": fraccion_densidad,
                    "densidad_valor": valor_densidad,
                    "riesgo": fraccion_riesgo,
                    "riesgo_valor": valor_riesgo,
                    # si se midió entropía, se loguea; si no, va None
                    "pasa_pureza": pasa_pureza,
                    "pasa_densidad": pasa_densidad,
                    "pasa_riesgo": pasa_riesgo,
                    "vecinos_validos_por_percentil": cant_dentro_densidad,
                    "synthetics_from_this_seed": int(
                        conteo_sinteticas_por_semilla[indice_local_semilla]
                    ),
                    "cant_vecinos_en_p_elegido": cant_dentro_densidad,
                    "cant_min_en_p_elegido": cant_min_dentro_densidad,
                }
            )


    # ----------------------------------------------------------------------
    # EXPORTAR LOG POR MUESTRAS A EXCEL
    # ----------------------------------------------------------------------
    def exportar_log_muestras_excel(self, ruta_excel, append=True):
        """
        Exporta self.logs_por_muestra a un archivo Excel.

        Parámetros:
        - ruta_excel: str o Path. Ruta completa al .xlsx
          (ej: "../datasets/datasets_aumentados/logs/pcsmote/por_muestras/log_pcsmote_x_muestra_ecoli.xlsx")
        - append:
            - True: si el archivo existe, lo lee, concatena filas nuevas y sobrescribe.
            - False: ignora el contenido anterior y sobrescribe con los logs actuales.

        Notas:
        - Si logs_por_muestra está vacío, no hace nada.
        """
        if not self.logs_por_muestra:
            # Nada que exportar
            return

        ruta_excel = Path(ruta_excel)
        ruta_excel.parent.mkdir(parents=True, exist_ok=True)

        df_nuevo = pd.DataFrame(self.logs_por_muestra)

        if append and ruta_excel.exists():
            try:
                df_existente = pd.read_excel(ruta_excel)
                df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
            except Exception:
                # Si falla la lectura, por seguridad se escribe solo lo nuevo
                df_final = df_nuevo
        else:
            df_final = df_nuevo

        df_final.to_excel(ruta_excel, index=False)


    # ----------------------------------------------------------------------
    # NUEVO: limpiar logs en memoria
    # ----------------------------------------------------------------------
    def limpiar_logs_por_muestra(self):
        """
        Vacía self.logs_por_muestra.
        Útil si querés liberar memoria después de volcar un bloque.
        """
        self.logs_por_muestra = []


    # ----------------------------------------------------------------------
    # NUEVO: exportar logs por muestra a CSV (append REAL, rápido)
    # ----------------------------------------------------------------------
    def exportar_log_muestras_csv(self, ruta_csv, append=True, sep=",", encoding="utf-8"):
        """
        Exporta self.logs_por_muestra a un archivo CSV.

        Parámetros:
        - ruta_csv: str o Path. Ruta completa al .csv
        - append:
            - True: agrega al final (modo append) sin re-leer el archivo.
            - False: sobrescribe (modo write).
        - sep: separador CSV (default ',')
        - encoding: encoding del archivo

        Notas:
        - Si el archivo no existe y append=True, escribe header.
        - Si logs_por_muestra está vacío, no hace nada.
        """
        if not self.logs_por_muestra:
            return

        ruta_csv = Path(ruta_csv)
        ruta_csv.parent.mkdir(parents=True, exist_ok=True)

        df_nuevo = pd.DataFrame(self.logs_por_muestra)

        existe = ruta_csv.exists()
        modo = "a" if append else "w"
        escribir_header = (not existe) or (not append)

        df_nuevo.to_csv(
            ruta_csv,
            mode=modo,
            index=False,
            header=escribir_header,
            sep=sep,
            encoding=encoding,
        )


    # ----------------------------------------------------------------------
    # NUEVO: exportar logs por muestra a CSV.GZ (compacto)
    # ----------------------------------------------------------------------
    def exportar_log_muestras_csv_gz(self, ruta_csv_gz, append=True, sep=",", encoding="utf-8"):
        """
        Igual que exportar_log_muestras_csv pero comprimido con gzip (.csv.gz).

        Importante:
        - Append en gzip funciona, pero cada append agrega un nuevo "member" gzip.
          Pandas puede leerlo igual (read_csv(compression="gzip")), pero el archivo
          no queda como un único stream "limpio". Aun así, para logs va perfecto.

        Si preferís cero sorpresas, usá CSV normal y comprimís al final.
        """
        if not self.logs_por_muestra:
            return

        ruta_csv_gz = Path(ruta_csv_gz)
        ruta_csv_gz.parent.mkdir(parents=True, exist_ok=True)

        df_nuevo = pd.DataFrame(self.logs_por_muestra)

        existe = ruta_csv_gz.exists()
        modo = "a" if append else "w"
        escribir_header = (not existe) or (not append)

        df_nuevo.to_csv(
            ruta_csv_gz,
            mode=modo,
            index=False,
            header=escribir_header,
            sep=sep,
            encoding=encoding,
            compression="gzip",
        )


    # ----------------------------------------------------------------------
    # NUEVO: convertir CSV -> XLSX al final (una sola vez)
    # ----------------------------------------------------------------------
    def convertir_csv_a_excel(self, ruta_csv, ruta_excel, sep=",", encoding="utf-8"):
        """
        Lee un CSV y lo exporta a Excel (.xlsx).
        Ideal para usar al final del proceso.

        Parámetros:
        - ruta_csv: str o Path al CSV
        - ruta_excel: str o Path al XLSX
        """
        ruta_csv = Path(ruta_csv)
        ruta_excel = Path(ruta_excel)
        ruta_excel.parent.mkdir(parents=True, exist_ok=True)

        if not ruta_csv.exists():
            return

        df = pd.read_csv(ruta_csv, sep=sep, encoding=encoding)
        df.to_excel(ruta_excel, index=False)

    # ----------------------------------------------------------------------
    # NUEVO: borrar archivo de log si existe
    # ----------------------------------------------------------------------
    def borrar_archivo_log(self, ruta):
        """
        Elimina un archivo de log si existe.

        Parámetros
        ----------
        ruta : str o Path
            Ruta completa al archivo (csv, csv.gz, xlsx, etc.)

        Retorna
        -------
        bool
            True  -> el archivo existía y fue borrado
            False -> el archivo no existía
        """
        ruta = Path(ruta)

        if ruta.exists():
            try:
                ruta.unlink()
                return True
            except Exception as e:
                raise RuntimeError(f"No se pudo borrar el archivo de log: {ruta}") from e

        return False
