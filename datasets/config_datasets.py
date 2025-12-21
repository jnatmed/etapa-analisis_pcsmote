from esquemas_conocidos import ESQUEMAS_CONOCIDOS


config_datasets = {
    # ─────────────────────────────1) US CRIME ─────────────────────────────
    "us_crime": {
        "path": "../datasets/US Crime/x10data.npz",
        "dataset_name": "us_crime",

        # Binario. Minoritaria identificada desde el benchmark imbalanced-learn (~12:1)
        "clase_minoria": 1,

        # El .npz trae X e y ya separados → el cargador detectará esto
        # Pero igual mantenemos col_target y col_features por consistencia
        "col_target": "target",
        "col_features": None,          # se completan luego de cargar X.shape[1]

        "sep": None,                   # no aplica (NPZ)
        "header": None,
        "binarizar": False,
        "tipo": "tabular_npz",         # IMPORTANTE: para que cargar_dataset.py lo trate distinto

        # US Crime no tiene outliers “físicos”; no inventes criterios
        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                # Nivel 1: sin cortes (no hay rangos físicos reales)
                "nivel_1": {
                    "tipo": "rango_fisico",
                    "criterios": {},
                    "fail_safe_max_ratio_eliminados": 0.0
                },
                # Nivel 2: marcar IQR por clase para diagnóstico
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                # Nivel 3: Isolation Forest apagado (evita borrar patrones útiles)
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "No eliminar filas; marcar por IQR si hace falta inspección."
        },

        "transformacion": {
            "escalado": {"tipo": "robust", "aplicar": True}
        }
    },
    # ─────────────────────────────2) SHUTTLE ─────────────────────────────
    "shuttle": {
        "path": "../datasets/statlog+shuttle/shuttle.csv",
        "dataset_name": "shuttle",
        "clase_minoria": 6,
        "clases_minor": [2, 6, 7],
        "col_features": ["time","A2","A3","A4","A5","A6","A7","A8","A9"],
        "col_target": "target",
        "sep": ",",
        "header": 0,
        "binarizar": False,
        "tipo": "tabular",
        # Limpieza: NO eliminar. Dataset ultra desbalanceado; extremos suelen ser clases minoritarias.
        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                # Nivel 1: solo rangos físicos triviales (>=0) para evitar errores de carga
                "nivel_1": {
                "tipo": "rango_fisico",
                "criterios": {"time": {"min": 0}, "A2": {"min": 0}, "A3": {"min": 0}, "A4": {"min": 0},
                                "A5": {"min": 0}, "A6": {"min": 0}, "A7": {"min": 0}, "A8": {"min": 0}, "A9": {"min": 0}},
                "fail_safe_max_ratio_eliminados": 0.0  # Shuttle: prohibido borrar
                },
                # Nivel 2: IQR POR CLASE solo para marcar (no borrar)
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                # Nivel 3: desactivado (evita que IF/OCSVM saquen minorías reales)
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "No eliminar outliers: las colas corresponden a clases 2/6/7. Solo marcar outliers por clase para diagnóstico."
        },
        "transformacion": {
            "escalado": {"tipo": "robust", "aplicar": True}
        }        
    },

    # ─────────────────────────────3) WDBC ─────────────────────────────
    "wdbc": {
        "path": "../datasets/breast+cancer+wisconsin+original/wdbc.data",
        "dataset_name": "wdbc",
        "clase_minoria": "M",
        "col_target": "diagnosis",
        "col_features": ESQUEMAS_CONOCIDOS["wdbc"][2:],  # 30 features
        "sep": ",",
        "header": None,
        "binarizar": False,
        "tipo": "tabular",
        "esquema": ESQUEMAS_CONOCIDOS["wdbc"],
        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                "nivel_1": {"tipo": "rango_fisico", "criterios": {}},   # sin cortes duros
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "Conservar extremos (malignos). Solo marcar por IQR si hace falta inspección."
        },
        "transformacion": { "escalado": {"tipo": "robust", "aplicar": True}
    }
    },

    # ─────────────────────────────4) GLASS ─────────────────────────────
    "glass": {
        "path": "../datasets/glass+identification/glass.data",
        "dataset_name": "glass",
        "clase_minoria": 6,
        "col_target": "Type",
        "col_features": ESQUEMAS_CONOCIDOS["glass"][1:-1],
        "sep": ",",
        "header": None,
        "binarizar": False,
        "tipo": "tabular",
        "esquema": ESQUEMAS_CONOCIDOS["glass"],
        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                # Rango químico plausible (quirúrgico)
                "nivel_1": {
                "tipo": "rango_fisico",
                "criterios": {"Si": {"min": 68, "max": 78}, "Ca": {"min": 4, "max": 15}, "K": {"max": 6}, "Fe": {"max": 1.0}},
                "fail_safe_max_ratio_eliminados": 0.02  # 2% permite sacar esas 3 filas (~1.4%)
                },
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "No tocar extremos de Ba/Mg/Al (discriminantes de headlamps/vajilla)."
        },
        "transformacion": {
            "winsorizacion": {"aplicar": True, "p_inferior": 0.01, "p_superior": 0.99},
            "escalado": {"tipo": "robust", "aplicar": True}
        }        
    },

    # ─────────────────────────────5) HEART ─────────────────────────────
    "heart": {
        "path": "../datasets/heart+disease/processed.cleveland.data",
        "dataset_name": "heart",
        "clase_minoria": 4,
        "col_target": "target",
        "col_features": ESQUEMAS_CONOCIDOS["heart"][:-1],
        "sep": ",",
        "header": None,
        "binarizar": False,
        "tipo": "tabular",
        "esquema": ESQUEMAS_CONOCIDOS["heart"],
        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                "nivel_1": {"tipo": "rango_fisico",
                            "criterios": {"trestbps": {"min": 80, "max": 200},
                                          "chol":     {"min": 100, "max": 400},
                                          "thalach":  {"min": 60, "max": 210},
                                          "oldpeak":  {"max": 5}}},
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "Eliminar solo valores fisiológicamente imposibles; el resto se marca."
        }
    },

    # ─────────────────────────────6) ECOLI ─────────────────────────────
    "ecoli": {
        "path": "../datasets/ecoli/ecoli.data",
        "dataset_name": "ecoli",
        "clase_minoria": "imL",
        "col_target": "class",
        "col_features": ESQUEMAS_CONOCIDOS["ecoli"][1:-1],
        "sep": r"\s+",
        "header": None,
        "binarizar": False,
        "tipo": "tabular",
        "esquema": ESQUEMAS_CONOCIDOS["ecoli"],
        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                "nivel_1": {"tipo": "rango_fisico",
                            "criterios": {"mcg": {"min": 0, "max": 1},
                                          "gvh": {"min": 0, "max": 1},
                                          "lip": {"min": 0, "max": 1},
                                          "chg": {"min": 0, "max": 1},
                                          "aac": {"min": 0, "max": 1},
                                          "alm1": {"min": 0, "max": 1},
                                          "alm2": {"min": 0, "max": 1}}},
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest",
                            "activar": True,
                            "params": {"contamination": 0.03, "n_estimators": 150, "random_state": 42},
                            "estrategia": "fit_por_clase"}
            },
            "comentario": "Valores normalizados [0,1]. Marcar por clase; IF leve solo para señal rara, no para borrar masivo."
        }
    },

    # ─────────────────────────────7) PREDICT_FAULTS ─────────────────────────────
    "predict_faults": {
        "path": "../datasets/predict_faults/predictive_maintenance.csv",
        "dataset_name": "predict_faults",

        "clase_minoria": "Random Failures",              # multiclase real
        "col_target": "Failure Type",

        # SOLO columnas numéricas útiles para el modelo
        "col_features": [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ],

        "sep": ",",
        "header": 0,
        "binarizar": False,
        "tipo": "tabular",

        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                "nivel_1": {
                    "tipo": "rango_fisico",
                    "criterios": {},
                    "fail_safe_max_ratio_eliminados": 0.0
                },
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "No eliminar outliers: las colas representan fallas reales."
        },

        "transformacion": {
            "escalado": {"tipo": "robust", "aplicar": True}
        }
    },

    # ─────────────────────────────8) GEAR VIBRATION ─────────────────────────────
    "gear_vibration": {
        "path": "../datasets/gear_vibration/gear_vibration_operativo.csv",
        "dataset_name": "gear_vibration",

        "clase_minoria": "root_crack",
        "col_target": "label",

        "col_features": [
            "s1_media",
            "s1_std",
            "s1_rms",
            "s2_media",
            "s2_std",
            "s2_rms",
            "s1_s2_corr",
            "speedSet",
            "load_value"
        ],

        "sep": ",",
        "header": 0,
        "binarizar": False,
        "tipo": "tabular",

        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                "nivel_1": {
                    "tipo": "rango_fisico",
                    "criterios": {},
                    "fail_safe_max_ratio_eliminados": 0.0
                },
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "Conservar extremos: pueden corresponder a patrones de falla."
        },

        "transformacion": {
            "escalado": {"tipo": "robust", "aplicar": True}
        }
    },

        # ───────────────────────────── TELCO CUSTOMER CHURN ─────────────────────────────

    # ─────────────────────────────9) TELCO CHURN ─────────────────────────────
    "telco_churn": {
        "path": "../datasets/telco_costumer_churn/telco_churn_analizable.csv",
        "dataset_name": "telco_costumer_churn",
        "clase_minoria": "churn",
        "col_target": "label",
# gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,PaperlessBilling,MonthlyCharges,TotalCharges,MultipleLines_No phone service,MultipleLines_Yes,InternetService_Fiber optic,InternetService_No,OnlineSecurity_No internet service,OnlineSecurity_Yes,OnlineBackup_No internet service,OnlineBackup_Yes,DeviceProtection_No internet service,DeviceProtection_Yes,TechSupport_No internet service,TechSupport_Yes,StreamingTV_No internet service,StreamingTV_Yes,StreamingMovies_No internet service,StreamingMovies_Yes,Contract_One year,Contract_Two year,PaymentMethod_Credit card (automatic),PaymentMethod_Electronic check,PaymentMethod_Mailed check        # 
        "col_features": [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "PaperlessBilling",
            "MonthlyCharges",
            "TotalCharges",
            "MultipleLines_No phone service",
            "MultipleLines_Yes",
            "InternetService_Fiber optic",
            "InternetService_No",
            "OnlineSecurity_No internet service",
            "OnlineSecurity_Yes",
            "OnlineBackup_No internet service",
            "OnlineBackup_Yes",
            "DeviceProtection_No internet service",
            "DeviceProtection_Yes",
            "TechSupport_No internet service",
            "TechSupport_Yes",
            "StreamingTV_No internet service",
            "StreamingTV_Yes",
            "StreamingMovies_No internet service",
            "StreamingMovies_Yes",
            "Contract_One year",
            "Contract_Two year",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check"
        ],
        "sep": ",",
        "header": 0,
        "binarizar": False,
        "tipo": "tabular",
        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                "nivel_1": {"tipo": "rango_fisico", "criterios": {}, "fail_safe_max_ratio_eliminados": 0.0},
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest", "activar": False},
            },
            "comentario": "No eliminar filas; solo marcar por IQR si se requiere diagnóstico."
        },
        "transformacion": {"escalado": {"tipo": "robust", "aplicar": True}},
    },


}
