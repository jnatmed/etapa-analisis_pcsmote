# esquemas_conocidos.py

ESQUEMAS_CONOCIDOS = {
    # ── SHUTTLE: 9 features + target ────────────────────────────────────────────
    # Usalo solo si leés SIN header (header=None). Si tu CSV tiene encabezado real,
    # dejá header=0 y NO se aplicará este esquema (está bien).
    "shuttle": [
        "A1","A2","A3","A4","A5","A6","A7","A8","A9","target"
    ],

    # ── WDBC (Breast Cancer Wisconsin, 'wdbc.data') ─────────────────────────────
    # 32 columnas: id, diagnosis + 30 features en el orden UCI.
    "wdbc": [
        "id","diagnosis",
        "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
        "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
        "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
    ],

    # ── GLASS (glass.data) ──────────────────────────────────────────────────────
    # 11 columnas: Id, 9 features, Type (target).
    "glass": [
        "Id","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"
    ],

    # ── HEART (Cleveland, processed.cleveland.data) ─────────────────────────────
    # 14 columnas. La original llama 'num' a la última; acá la nombramos 'target'
    # para que coincida con tu config_datasets.
    "heart": [
        "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
        "oldpeak","slope","ca","thal","target"
    ],

    # ── IRIS (iris.data) ────────────────────────────────────────────────────────
    # 5 columnas: 4 features + class.
    "iris": [
        "sepal_length","sepal_width","petal_length","petal_width","class"
    ],

    # ── ECOLI (ecoli.data) ──────────────────────────────────────────────────────
    # 9 columnas: un id/string, 7 features, class. Tu config usa solo las 7 features.
    "ecoli": [
        "seq_name","mcg","gvh","lip","chg","aac","alm1","alm2","class"
    ],
}
