# ===================== Clase: GraficadorResultados =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import os

class GraficadorResultados:
    """
    Clase utilitaria para normalizar, resumir y graficar resultados de experimentos
    con y sin IsolationForest, incluyendo datasets base y aumentados (PC-SMOTE).
    
    Flujo típico:
      1) instanciar con dos DataFrames: df_sin_iso y df_con_iso
      2) llamar a preparar_datos() para unificar y calcular columnas derivadas
      3) usar tablas_* y graficar_* para documentar resultados
    """

    def __init__(self, df_sin_iso: pd.DataFrame, df_con_iso: pd.DataFrame):
        """
        Args:
            df_sin_iso: DataFrame con resultados sin IsolationForest.
            df_con_iso: DataFrame con resultados con IsolationForest.
        """
        self.df_sin_iso = df_sin_iso.copy()
        self.df_con_iso = df_con_iso.copy()
        self.df_unificado = None  # se construye en preparar_datos()

    def reporte_graficos_completo(self,
                                dataset: Optional[str] = None,
                                modelo: Optional[str] = None,
                                metrica: str = "test_f1_macro",
                                usar_delta_heatmap: bool = False,
                                guardar_prefix: Optional[str] = None) -> None:
        """
        Genera un set estándar de gráficos para documentar resultados.
        Parámetros:
        - dataset: filtra por dataset (None = todos).
        - modelo:  filtra por modelo en los gráficos que lo soportan (líneas por técnica / CV vs Test).
        - metrica: métrica a usar en heatmaps/scatter (por defecto 'test_f1_macro').
        - usar_delta_heatmap: si True, el heatmap usa ΔF1 (Test − CV).
        - guardar_prefix: si no es None, guarda cada figura como PNG con este prefijo.
                            Ej: 'figs/glass' → 'figs/glass_f1_linea.png', etc.
        """

        self._asegurar_unificado()

        def _guardar(fig_name: str):
            if guardar_prefix:
                os.makedirs(os.path.dirname(guardar_prefix), exist_ok=True)
                plt.savefig(f"{guardar_prefix}_{fig_name}.png", dpi=150, bbox_inches="tight")

        # 1) Línea: F1(Test) por técnica (con/sin isolation)
        self.graficar_f1_test_por_tecnica(dataset=dataset, modelo=modelo)
        _guardar("f1_linea")
        plt.close()

        # 2) Línea: ΔF1 por técnica (con/sin isolation)
        self.graficar_delta_f1_por_tecnica(dataset=dataset, modelo=modelo)
        _guardar("deltaf1_linea")
        plt.close()

        # 3) Heatmap: modelo × técnica (F1 o ΔF1)
        self.graficar_heatmap_modelo_tecnica(dataset=dataset,
                                            metrica=metrica,
                                            usar_delta=usar_delta_heatmap)
        _guardar("heatmap_modelo_tecnica")
        plt.close()

        # 4) Dispersión: CV vs Test (mide sobreajuste)
        self.graficar_cv_vs_test(dataset=dataset, modelo=modelo)
        _guardar("cv_vs_test_scatter")
        plt.close()

        # 5) Boxplot: distribución ΔF1 por técnica
        self.graficar_boxplot_delta(dataset=dataset)
        _guardar("deltaf1_boxplot")
        plt.close()


    # -------------------- Gráficos de boxplot --------------------
    def graficar_boxplot_delta(self, dataset: Optional[str] = None, ancho=8, alto=5):
        """
        Boxplot del ΔF1(Test−CV) por técnica.
        """
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset:
            df = df[df["dataset"] == dataset]

        plt.figure(figsize=(ancho, alto))
        df.boxplot(column="delta_f1", by="tecnica_compuesta", grid=False)
        plt.title(f"Distribución ΔF1(Test−CV){' · ' + dataset if dataset else ''}")
        plt.suptitle("")
        plt.ylabel("Δ F1")
        plt.axhline(0, linestyle="--", color="gray")
        plt.tight_layout()
        plt.show()


    # -------------------- Gráficos de dispersión --------------------
    def graficar_cv_vs_test(self, dataset: str | None = None,
                            modelo: str | None = None,
                            separar_por_tecnica: bool = False,
                            ancho: int = 6, alto: int = 6) -> None:
        """
        Dispersión CV vs Test (F1 macro) con leyenda:
        - Color/serie: con/sin Isolation
        - Opcional: marcadores distintos por técnica (separar_por_tecnica=True)
        """
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset:
            df = df[df["dataset"] == dataset]
        if modelo:
            df = df[df["modelo"] == modelo]

        plt.figure(figsize=(ancho, alto))

        # una serie por con_isolation (dos colores por defecto) con etiquetas para la leyenda
        for iso_flag, datos in df.groupby("con_isolation"):
            etiqueta_iso = "Con Isolation" if iso_flag else "Sin Isolation"

            if separar_por_tecnica:
                # marcador distinto por técnica (mismos colores por iso)
                marcadores = ["o", "s", "D", "^", "v", "P", "X"]
                tecnicas = self._orden_tecnicas(datos["tecnica_compuesta"].unique())
                for i, (tec, sub) in enumerate(datos.groupby("tecnica_compuesta")):
                    plt.scatter(sub["cv_f1_macro"], sub["test_f1_macro"],
                                marker=marcadores[i % len(marcadores)],
                                label=f"{etiqueta_iso} · {tec}",
                                alpha=0.85)
            else:
                plt.scatter(datos["cv_f1_macro"], datos["test_f1_macro"],
                            label=etiqueta_iso, alpha=0.85)

        # diagonal ideal
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

        plt.xlabel("F1 (CV)")
        plt.ylabel("F1 (Test)")
        titulo = "CV vs Test F1"
        if dataset: titulo += f" · dataset={dataset}"
        if modelo:  titulo += f" · modelo={modelo}"
        plt.title(titulo)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best", title="Series")
        plt.tight_layout()
        plt.show()

    # -------------------- Gráficos de heatmap --------------------
    def graficar_heatmap_modelo_tecnica(self, dataset: Optional[str] = None,
                                    metrica: str = "test_f1_macro",
                                    usar_delta: bool = False,
                                    ancho: int = 8,
                                    alto: int = 5) -> None:
        """
        Heatmap de F1(Test) o ΔF1 por modelo y técnica.
        Si usar_delta=True, grafica ΔF1(Test−CV).
        """
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset:
            df = df[df["dataset"] == dataset]

        metrica_real = "delta_f1" if usar_delta else metrica
        if metrica_real not in df.columns:
            raise ValueError(f"La métrica '{metrica_real}' no está disponible.")

        pivot = (df.groupby(["modelo", "tecnica_compuesta"], dropna=False)
                [metrica_real].mean().unstack())

        plt.figure(figsize=(ancho, alto))
        plt.imshow(pivot, cmap="YlGnBu", aspect="auto")
        plt.colorbar(label=f"{'Δ ' if usar_delta else ''}{metrica}")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=15)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        titulo = f"Heatmap {metrica_real}"
        if dataset: titulo += f" · dataset={dataset}"
        plt.title(titulo)
        for i, modelo in enumerate(pivot.index):
            for j, tecnica in enumerate(pivot.columns):
                valor = pivot.loc[modelo, tecnica]
                if not np.isnan(valor):
                    plt.text(j, i, f"{valor:.2f}", ha="center", va="center", color="black")
        plt.tight_layout()
        plt.show()


    # -------------------- Normalización y unión --------------------

    def _columna_tecnica_compuesta(self, df: pd.DataFrame) -> pd.Series:
        """
        Construye una etiqueta de técnica legible combinando:
        - tipo_combination (base/aumentado)
        - tecnica_aumento (base/pcsmote)
        - criterio_pureza (entropia/proporcion/--) cuando aplique

        Devuelve una serie con valores tipo:
          'base'
          'pcsmote (proporcion)'
          'pcsmote (entropia)'
        """
        def _rotulo(row) -> str:
            tc = str(row.get("tipo_combination", "")).strip().lower()
            ta = str(row.get("tecnica_aumento", "")).strip().lower()
            cp = str(row.get("criterio_pureza", "")).strip().lower()

            if tc == "base" or ta == "base" or ta == "--":
                return "base"
            if ta == "pcsmote":
                suf = cp if cp not in ("", "--", "nan") else ""
                return f"pcsmote ({suf})".strip()
            return ta if ta else tc

        return df.apply(_rotulo, axis=1)

    def preparar_datos(self) -> pd.DataFrame:
        """
        Unifica df_sin_iso y df_con_iso, agrega bandera 'con_isolation' y construye:
        - tecnica_compuesta (etiqueta legible)
        - delta_f1 = test_f1_macro - cv_f1_macro
        (lo mismo puede extenderse a otras métricas si lo necesitás)

        Returns:
            DataFrame unificado listo para tablas y gráficos.
        """
        a = self.df_sin_iso.copy()
        b = self.df_con_iso.copy()

        a["con_isolation"] = False
        b["con_isolation"] = True

        # Etiqueta de técnica legible y robusta
        a["tecnica_compuesta"] = self._columna_tecnica_compuesta(a)
        b["tecnica_compuesta"] = self._columna_tecnica_compuesta(b)

        df = pd.concat([a, b], ignore_index=True)

        # Campos esperados; si faltan, no rompe
        for col in ["cv_f1_macro", "test_f1_macro"]:
            if col not in df.columns:
                df[col] = np.nan

        # Delta F1 (Test - CV)
        df["delta_f1"] = df["test_f1_macro"] - df["cv_f1_macro"]

        # Normalizar algunos nombres clave para filtros
        df["dataset"] = df["dataset_logico"].astype(str)
        df["modelo"]  = df["nombre_modelo_aprendizaje"].astype(str)

        # Orden sugerido para técnicas en gráficos/tablas
        df["tecnica_compuesta"] = df["tecnica_compuesta"].replace({
            "pcsmote (proporcion)": "pcsmote (proporción)"
        })

        self.df_unificado = df
        return df

    # -------------------- Tablas resumen --------------------

    def tabla_resumen_por_dataset_modelo(self) -> pd.DataFrame:
        """
        Promedios por (dataset, modelo, tecnica_compuesta, con_isolation) de:
           cv_f1_macro, test_f1_macro, delta_f1
        """
        self._asegurar_unificado()
        cols = ["dataset", "modelo", "tecnica_compuesta", "con_isolation"]
        agregados = {
            "cv_f1_macro": "mean",
            "test_f1_macro": "mean",
            "delta_f1": "mean",
            "cantidad_train": "mean",
            "cantidad_test": "mean",
            "cantidad_caracteristicas": "mean",
        }
        # solo agrega si existen
        agregados = {k: v for k, v in agregados.items() if k in self.df_unificado.columns}

        tabla = (self.df_unificado
                 .groupby(cols, dropna=False)
                 .agg(agregados)
                 .reset_index()
                 .sort_values(["dataset", "modelo", "tecnica_compuesta", "con_isolation"]))
        return tabla

    def tabla_mejor_tecnica_por_modelo(self, por: str = "test_f1_macro") -> pd.DataFrame:
        """
        Selecciona por dataset y modelo la mejor técnica según 'por' (por defecto F1 Test).
        """
        self._asegurar_unificado()
        df = self.tabla_resumen_por_dataset_modelo()
        if por not in df.columns:
            raise ValueError(f"La métrica '{por}' no está en la tabla resumen.")
        idx = df.groupby(["dataset", "modelo"])[por].idxmax()
        return df.loc[idx].reset_index(drop=True)

    # -------------------- Gráficos --------------------

    def graficar_f1_test_por_tecnica(self,
                                    dataset: Optional[str] = None,
                                    modelo: Optional[str] = None,
                                    ancho: int = 10,
                                    alto: int = 5) -> None:
        """
        Gráfico de línea del F1(Test) promediado por técnica y bandera con_isolation.
        Filtros opcionales por dataset y/o modelo.
        """
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset:
            df = df[df["dataset"] == dataset]
        if modelo:
            df = df[df["modelo"] == modelo]

        g = (df.groupby(["tecnica_compuesta", "con_isolation"], dropna=False)
                [["test_f1_macro"]].mean().reset_index())

        orden_tecnicas = self._orden_tecnicas(g["tecnica_compuesta"].unique())
        g["tecnica_compuesta"] = pd.Categorical(g["tecnica_compuesta"], categories=orden_tecnicas, ordered=True)
        g = g.sort_values(["tecnica_compuesta", "con_isolation"])

        plt.figure(figsize=(ancho, alto))

        for iso_flag, grupo in g.groupby("con_isolation"):
            etiqueta = "con Isolation" if iso_flag else "sin Isolation"
            plt.plot(grupo["tecnica_compuesta"], grupo["test_f1_macro"], 
                    marker='o', label=etiqueta)

        plt.ylabel("F1 macro (Test)")
        titulo = "F1(Test) por técnica"
        if dataset: titulo += f" · dataset={dataset}"
        if modelo:  titulo += f" · modelo={modelo}"
        plt.title(titulo)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()



    """
    Barras de F1(Test) por modelo, separando técnica y con/sin isolation.
    """
    def graficar_f1_test_por_modelo(self,
                                    dataset: Optional[str] = None,
                                    ancho: int = 11,
                                    alto: int = 5) -> None:
        """
        Gráfico de líneas del F1(Test) por modelo, separando técnica y con/sin isolation.
        """
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset:
            df = df[df["dataset"] == dataset]

        # Agrupar y calcular promedios
        g = (df.groupby(["modelo", "tecnica_compuesta", "con_isolation"], dropna=False)
                [["test_f1_macro"]].mean().reset_index())

        # Ordenar técnicas de forma consistente
        orden_tecnicas = self._orden_tecnicas(g["tecnica_compuesta"].unique())
        g["tecnica_compuesta"] = pd.Categorical(g["tecnica_compuesta"],
                                                categories=orden_tecnicas,
                                                ordered=True)

        plt.figure(figsize=(ancho, alto))

        # Una línea por modelo (cada una con dos sub-series: sin y con isolation)
        for modelo, datos_modelo in g.groupby("modelo"):
            for iso_flag, datos_iso in datos_modelo.groupby("con_isolation"):
                etiqueta = f"{modelo} - {'con Isolation' if iso_flag else 'sin Isolation'}"
                plt.plot(datos_iso["tecnica_compuesta"],
                        datos_iso["test_f1_macro"],
                        marker='o',
                        label=etiqueta)

        plt.ylabel("F1 macro (Test)")
        titulo = "F1(Test) por modelo y técnica"
        if dataset:
            titulo += f" · dataset={dataset}"
        plt.title(titulo)
        plt.xticks(rotation=20)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


    """
    Gráfico de línea de ΔF1 (Test−CV) promediado por técnica y con_isolation.
    """
    def graficar_delta_f1_por_tecnica(self,
                                    dataset: Optional[str] = None,
                                    modelo: Optional[str] = None,
                                    ancho: int = 10,
                                    alto: int = 5) -> None:
        """
        Gráfico de línea de ΔF1 (Test−CV) promediado por técnica y con_isolation.
        """
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset:
            df = df[df["dataset"] == dataset]
        if modelo:
            df = df[df["modelo"] == modelo]

        g = (df.groupby(["tecnica_compuesta", "con_isolation"], dropna=False)
                [["delta_f1"]].mean().reset_index())

        orden_tecnicas = self._orden_tecnicas(g["tecnica_compuesta"].unique())
        g["tecnica_compuesta"] = pd.Categorical(g["tecnica_compuesta"], categories=orden_tecnicas, ordered=True)
        g = g.sort_values(["tecnica_compuesta", "con_isolation"])

        plt.figure(figsize=(ancho, alto))

        for iso_flag, grupo in g.groupby("con_isolation"):
            etiqueta = "con Isolation" if iso_flag else "sin Isolation"
            plt.plot(grupo["tecnica_compuesta"], grupo["delta_f1"], 
                    marker='o', label=etiqueta)

        plt.ylabel("Δ F1 (Test − CV)")
        titulo = "Δ F1 por técnica"
        if dataset: titulo += f" · dataset={dataset}"
        if modelo:  titulo += f" · modelo={modelo}"
        plt.title(titulo)
        plt.axhline(0.0, linestyle="--", color="gray", alpha=0.6)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


    # -------------------- Utilitarios internos --------------------

    def _asegurar_unificado(self) -> None:
        """Verifica que preparar_datos() se haya ejecutado."""
        if self.df_unificado is None:
            raise RuntimeError("Debes llamar a preparar_datos() antes de solicitar tablas o gráficos.")

    @staticmethod
    def _orden_tecnicas(valores: List[str]) -> List[str]:
        """Devuelve un orden consistente: base primero, luego pcsmote (proporción), luego pcsmote (entropía)."""
        preferencia = ["base", "pcsmote (proporción)", "pcsmote (entropia)", "pcsmote (entropía)"]
        en_lista = [v for v in preferencia if v in valores]
        restantes = [v for v in valores if v not in en_lista]
        return en_lista + restantes

    # -------------------- Exportación opcional --------------------

    def exportar_resumen_csv(self, ruta_salida: str) -> None:
        """Exporta la tabla resumen principal a CSV."""
        tabla = self.tabla_resumen_por_dataset_modelo()
        tabla.to_csv(ruta_salida, index=False)


    # ---------- helpers que dibujan en un Axes (no crean figuras) ----------
    def _plot_f1_linea_ax(self, ax, dataset=None, modelo=None):
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset: df = df[df["dataset"] == dataset]
        if modelo:  df = df[df["modelo"] == modelo]
        g = (df.groupby(["tecnica_compuesta", "con_isolation"], dropna=False)
            [["test_f1_macro"]].mean().reset_index())
        orden = self._orden_tecnicas(g["tecnica_compuesta"].unique())
        g["tecnica_compuesta"] = pd.Categorical(g["tecnica_compuesta"], categories=orden, ordered=True)
        g = g.sort_values(["tecnica_compuesta","con_isolation"])
        for iso_flag, grupo in g.groupby("con_isolation"):
            ax.plot(grupo["tecnica_compuesta"], grupo["test_f1_macro"], marker="o",
                    label=("con Isolation" if iso_flag else "sin Isolation"))
        ax.set_title("F1(Test) por técnica")
        ax.set_ylabel("F1 macro (Test)")
        ax.grid(True, linestyle="--", alpha=.5)
        ax.legend()

    def _plot_delta_linea_ax(self, ax, dataset=None, modelo=None):
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset: df = df[df["dataset"] == dataset]
        if modelo:  df = df[df["modelo"] == modelo]
        g = (df.groupby(["tecnica_compuesta","con_isolation"], dropna=False)
            [["delta_f1"]].mean().reset_index())
        orden = self._orden_tecnicas(g["tecnica_compuesta"].unique())
        g["tecnica_compuesta"] = pd.Categorical(g["tecnica_compuesta"], categories=orden, ordered=True)
        g = g.sort_values(["tecnica_compuesta","con_isolation"])
        for iso_flag, grupo in g.groupby("con_isolation"):
            ax.plot(grupo["tecnica_compuesta"], grupo["delta_f1"], marker="o",
                    label=("con Isolation" if iso_flag else "sin Isolation"))
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("ΔF1 (Test−CV) por técnica")
        ax.set_ylabel("Δ F1")
        ax.grid(True, linestyle="--", alpha=.5)
        ax.legend()

    def _plot_heatmap_ax(self, ax, dataset=None, metrica="test_f1_macro", usar_delta=False):
        import numpy as np
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset: df = df[df["dataset"] == dataset]
        metrica_real = "delta_f1" if usar_delta else metrica
        pivot = (df.groupby(["modelo","tecnica_compuesta"], dropna=False)[metrica_real]
                .mean().unstack())
        im = ax.imshow(pivot, cmap="YlGnBu", aspect="auto")
        ax.set_title(f"Heatmap {metrica_real}")
        ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=15)
        ax.set_yticks(np.arange(len(pivot.index)));   ax.set_yticklabels(pivot.index)
        for i, m in enumerate(pivot.index):
            for j, t in enumerate(pivot.columns):
                val = pivot.loc[m, t]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
        return im  # para colorbar

    def _plot_scatter_cv_test_ax(self, ax, dataset=None, modelo=None, separar_por_tecnica=False):
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset: df = df[df["dataset"] == dataset]
        if modelo:  df = df[df["modelo"] == modelo]
        for iso_flag, datos in df.groupby("con_isolation"):
            etiqueta_iso = "Con Isolation" if iso_flag else "Sin Isolation"
            if separar_por_tecnica:
                marcadores = ["o","s","D","^","v","P","X"]
                for i, (tec, sub) in enumerate(datos.groupby("tecnica_compuesta")):
                    ax.scatter(sub["cv_f1_macro"], sub["test_f1_macro"],
                            marker=marcadores[i % len(marcadores)],
                            label=f"{etiqueta_iso} · {tec}", alpha=.85)
            else:
                ax.scatter(datos["cv_f1_macro"], datos["test_f1_macro"],
                        label=etiqueta_iso, alpha=.85)
        ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
        ax.set_title("CV vs Test F1"); ax.set_xlabel("F1 (CV)"); ax.set_ylabel("F1 (Test)")
        ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8)

    def _plot_boxplot_delta_ax(self, ax, dataset=None):
        self._asegurar_unificado()
        df = self.df_unificado.copy()
        if dataset: df = df[df["dataset"] == dataset]
        grupos = [g for _, g in df.groupby("tecnica_compuesta")]
        etiquetas = [str(n) for n, _ in df.groupby("tecnica_compuesta")]
        ax.boxplot([g["delta_f1"].values for g in grupos], labels=etiquetas)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("Distribución ΔF1 por técnica"); ax.set_ylabel("Δ F1")
        ax.grid(True, linestyle="--", alpha=.5)

    # ---------- figura compuesta 3×2 ----------
    def panel_por_dataset(self, dataset: str,
                        modelo: Optional[str] = None,
                        metrica_heatmap: str = "test_f1_macro",
                        usar_delta_heatmap: bool = False,
                        figsize: tuple = (14, 12),
                        guardar_path: Optional[str] = None):
        """
        Genera una única imagen con 5 gráficos para `dataset`:
        (1) F1(Test) línea, (2) ΔF1 línea, (3) Heatmap modelo×técnica,
        (4) CV vs Test scatter, (5) Boxplot ΔF1. Disposición 3×2.
        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        axs = axs.ravel()

        # 1) F1 línea
        self._plot_f1_linea_ax(axs[0], dataset=dataset, modelo=modelo)
        # 2) ΔF1 línea
        self._plot_delta_linea_ax(axs[1], dataset=dataset, modelo=modelo)
        # 3) Heatmap
        m = self._plot_heatmap_ax(axs[2], dataset=dataset,
                                metrica=metrica_heatmap, usar_delta=usar_delta_heatmap)
        fig.colorbar(m, ax=axs[2], fraction=0.046, pad=0.04)
        # 4) CV vs Test
        self._plot_scatter_cv_test_ax(axs[3], dataset=dataset, modelo=modelo, separar_por_tecnica=False)
        # 5) Boxplot ΔF1
        self._plot_boxplot_delta_ax(axs[4], dataset=dataset)
        # 6) panel vacío (oculto)
        axs[5].axis("off")

        fig.suptitle(f"Resumen de resultados · dataset={dataset}" + (f" · modelo={modelo}" if modelo else ""), y=0.995)
        plt.tight_layout()

        if guardar_path:
            os.makedirs(os.path.dirname(guardar_path), exist_ok=True)
            plt.savefig(guardar_path, dpi=150, bbox_inches="tight")
        plt.show()

# ===================== Fin clase: GraficadorResultados =====================
