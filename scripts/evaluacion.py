import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split

def evaluar_sampler_completo(nombre, sampler_class, X, y_bin, n_iter=10, **kwargs):
    metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "balanced_acc": []
    }

    for seed in range(n_iter):
        sampler = sampler_class(random_state=seed, **kwargs)
        X_res, y_res = sampler.fit_resample(X, y_bin)
        model = RandomForestClassifier(random_state=seed)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X_res, y_res):
            model.fit(X_res[train_idx], y_res[train_idx])
            y_pred = model.predict(X_res[test_idx])
            y_prob = model.predict_proba(X_res[test_idx])[:, 1]

            metrics["precision"].append(precision_score(y_res[test_idx], y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_res[test_idx], y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y_res[test_idx], y_pred, zero_division=0))
            metrics["roc_auc"].append(roc_auc_score(y_res[test_idx], y_prob))
            metrics["balanced_acc"].append(balanced_accuracy_score(y_res[test_idx], y_pred))

    return {
        "técnica": nombre,
        "mean_precision": np.mean(metrics["precision"]),
        "mean_recall": np.mean(metrics["recall"]),
        "mean_f1": np.mean(metrics["f1"]),
        "std_f1": np.std(metrics["f1"]),
        "mean_roc_auc": np.mean(metrics["roc_auc"]),
        "mean_bal_acc": np.mean(metrics["balanced_acc"])
    }

def evaluar_sampler_holdout(nombre, sampler_class, X, y_bin, n_iter=5, test_size=0.3, modelo=None, **kwargs):
    metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "balanced_acc": []
    }

    for seed in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_bin, test_size=test_size, stratify=y_bin, random_state=seed
        )

        sampler = sampler_class(random_state=seed, **kwargs)
        X_res, y_res = sampler.fit_resample(X_train, y_train)

        try:
            # ⚠️ Acá usás el modelo pasado, si no, RandomForest por defecto
            model = modelo(random_state=seed) if modelo is not None else RandomForestClassifier(random_state=seed)
        except TypeError:
            print(f"⚠️  Clasificador {modelo.__name__} no acepta random_state, se instancia sin él.")
            model = modelo()
            
        model.fit(X_res, y_res)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))
        metrics["balanced_acc"].append(balanced_accuracy_score(y_test, y_pred))

    return {
        "técnica": nombre,
        "mean_precision": np.mean(metrics["precision"]),
        "mean_recall": np.mean(metrics["recall"]),
        "mean_f1": np.mean(metrics["f1"]),
        "std_f1": np.std(metrics["f1"]),
        "mean_roc_auc": np.mean(metrics["roc_auc"]),
        "mean_bal_acc": np.mean(metrics["balanced_acc"])
    }
