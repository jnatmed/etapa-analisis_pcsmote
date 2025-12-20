# Estructura del proyecto

Este proyecto está organizado para facilitar la ejecución de experimentos con técnicas de sobremuestreo en clasificación de datos desbalanceados. A continuación se describe la estructura de carpetas utilizada:

## ✅ Estructura base

- `datasets/`: contiene los datos originales crudos utilizados para entrenar y evaluar los modelos. Esta carpeta no debe modificarse manualmente para garantizar la reproducibilidad.
- `notebooks/`: contiene notebooks de Jupyter donde se realizan los análisis exploratorios, pruebas de modelos, comparaciones y visualizaciones.
- `resultados/`: almacena las salidas generadas por los experimentos, como métricas, gráficos, logs, archivos `.csv` o `.txt`, entre otros.
- `README.md`: archivo de documentación principal del proyecto. Incluye descripción general, instrucciones de ejecución y dependencias necesarias.
- `scripts/`: scripts en Python reutilizables, como funciones auxiliares, lógica de preprocesamiento o entrenamiento.
- `models/`: modelos entrenados almacenados en formato `.pkl`, `.joblib`, etc.
- `figures/` o `plots/`: imágenes o visualizaciones generadas automáticamente desde los experimentos.
- `experiments/` o subcarpetas dentro de `notebooks/`: para separar los experimentos por técnica, dataset o fecha, facilitando el seguimiento de cada variante.

---

Este enfoque modular mejora la organización del código y los resultados, permite escalar el proyecto con mayor facilidad y contribuye a una documentación clara para futuras consultas o publicaciones.
