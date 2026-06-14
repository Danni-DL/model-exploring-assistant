"""
model_fit.py
============
Shared "model fit generator" logic for the CSP assessment. Imported by BOTH the
model-fit app and the result-presenter app, so the fitting lives in exactly one
place and the presenter can run it live from its variable multiselect.

Design stance (the interview talking points)
--------------------------------------------
* The tool does NOT decide which model is "correct." It defines a model-agnostic
  RESULTS CONTRACT (see `fit_and_report` return value) that any estimator can
  fill. We ship a deliberately simple demo estimator (logistic regression on
  selected variables -> diagnosis) only to populate that contract with realistic
  output. Swapping in another model means swapping the estimator, not the tool.
* Variance / CI come from the BOOTSTRAP, not from a model-specific formula. That
  is the point: bootstrap gives uncertainty on coefficients AND on metrics for
  any estimator, and it is honest about small-n (CIs come out wide, which is the
  correct message when n=30-60 and there is no agreed metric).
* Validation without an agreed metric -> we report several lenses, each with a
  CI: overall accuracy, per-class sensitivity (recall), the confusion matrix,
  and an explicit failure-case list. No single number is treated as ground.

This module has no Streamlit dependency.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix,
                             r2_score, mean_absolute_error, silhouette_score)


# Candidate predictor variables the apps expose in the multiselect.
EMB_PCS = ["emb_pc1", "emb_pc2", "emb_pc3", "emb_pc4", "emb_pc5"]
CONTINUOUS_VARS = EMB_PCS + ["age", "quality_score", "behavioral_score", "session"]
CATEGORICAL_VARS = ["site", "sex"]
ALL_CANDIDATES = CONTINUOUS_VARS + CATEGORICAL_VARS


# --------------------------------------------------------------------------- #
# Feature table: merge modalities to the (subject, session) grain
# --------------------------------------------------------------------------- #
def build_feature_table(clinical: pd.DataFrame, embeddings: pd.DataFrame,
                        ground_truth: pd.DataFrame, n_pcs: int = 5) -> pd.DataFrame:
    """One row per existing (subject, session). Embeddings are reduced to a few
    PCA components so a coefficient table is interpretable and the fit is stable
    at small n. Rows without an embedding keep NaN PCs (missingness preserved)."""
    df = clinical.copy()

    emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
    if len(embeddings) >= 2 and emb_cols:
        Xe = embeddings[emb_cols].to_numpy(dtype=float)
        Xe = StandardScaler().fit_transform(Xe)
        k = int(min(n_pcs, Xe.shape[1], Xe.shape[0] - 1))
        pcs = PCA(n_components=k).fit_transform(Xe)
        pcdf = embeddings[["subject_id", "session"]].copy().reset_index(drop=True)
        for i in range(k):
            pcdf[f"emb_pc{i + 1}"] = pcs[:, i]
        df = df.merge(pcdf, on=["subject_id", "session"], how="left")

    # ensure all candidate PC columns exist even if k < 5
    for c in EMB_PCS:
        if c not in df.columns:
            df[c] = np.nan

    if "true_diagnosis" in ground_truth.columns:
        df = df.merge(ground_truth[["subject_id", "true_diagnosis"]],
                      on="subject_id", how="left")
    return df


def available_candidates(feature_df: pd.DataFrame) -> list:
    """Candidates that exist and are not entirely missing in this dataset."""
    out = []
    for v in ALL_CANDIDATES:
        if v in feature_df.columns and feature_df[v].notna().any():
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Design matrix assembly (encode categoricals, standardize continuous)
# --------------------------------------------------------------------------- #
def _assemble(feature_df: pd.DataFrame, selected_vars: Sequence[str],
              target_col: Optional[str], impute: bool):
    cont = [v for v in selected_vars if v in CONTINUOUS_VARS]
    cat = [v for v in selected_vars if v in CATEGORICAL_VARS]

    parts, names = [], []
    if cont:
        parts.append(feature_df[cont].reset_index(drop=True))
        names += cont
    if cat:
        dummies = pd.get_dummies(feature_df[cat].astype("object"),
                                 drop_first=True, dummy_na=False)
        parts.append(dummies.reset_index(drop=True))
        names += list(dummies.columns)

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(
        index=range(len(feature_df)))
    meta = feature_df[["subject_id", "session", "site"]].reset_index(drop=True)
    if "true_diagnosis" in feature_df.columns:
        meta = meta.assign(true_diagnosis=feature_df["true_diagnosis"].values)

    y = (feature_df[target_col].reset_index(drop=True)
         if target_col is not None else None)

    if impute:
        for c in X.columns:
            X[c] = X[c].fillna(X[c].mean())
        mask = pd.Series(True, index=X.index)
    else:
        mask = X.notna().all(axis=1) if len(X.columns) else pd.Series(
            True, index=range(len(feature_df)))
    if y is not None:
        mask = mask & y.notna()

    dropped = int((~mask).sum())
    Xk = X[mask].to_numpy(dtype=float) if len(X.columns) else np.empty((mask.sum(), 0))
    metak = meta[mask.values].reset_index(drop=True)
    yk = y[mask].to_numpy() if y is not None else None

    # standardize continuous columns only (first len(cont) columns)
    if cont and Xk.shape[1] >= len(cont):
        Xk = Xk.copy()
        Xk[:, :len(cont)] = StandardScaler().fit_transform(Xk[:, :len(cont)])

    return Xk, yk, metak, names, dropped


def _percentile_ci(arr, lo=2.5, hi=97.5):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return (np.nan, np.nan)
    return (float(np.nanpercentile(arr, lo)), float(np.nanpercentile(arr, hi)))


# --------------------------------------------------------------------------- #
# CLASSIFICATION (the primary demo model used by the presenter)
# --------------------------------------------------------------------------- #
def fit_and_report(feature_df: pd.DataFrame, selected_vars: Sequence[str],
                   target_col: str = "diagnosis", n_boot: int = 200,
                   n_folds: int = 5, seed: int = 0, impute: bool = False,
                   C: float = 1.0) -> dict:
    """Fit logistic regression on `selected_vars` -> `target_col` and return the
    unified results contract. Variance/CI via bootstrap; predictions via
    out-of-fold cross-validation."""
    rng = np.random.default_rng(seed)

    if not selected_vars:
        return {"ok": False, "error": "No variables selected."}

    X, y, meta, feat_names, dropped = _assemble(feature_df, selected_vars,
                                                 target_col, impute)
    n = len(y) if y is not None else 0
    if n < 8 or X.shape[1] == 0:
        return {"ok": False, "error": f"Too few complete rows to fit "
                f"(n={n}, features={X.shape[1]}). Try fewer variables, enable "
                f"imputation, or generate more subjects."}

    classes = np.unique(y)
    if len(classes) < 2:
        return {"ok": False, "error": "Target has fewer than 2 classes after "
                "dropping missing rows."}

    counts = pd.Series(y).value_counts()
    min_class = int(counts.min())
    folds = max(2, min(n_folds, min_class))
    if min_class < 2:
        return {"ok": False, "error": f"Class '{counts.idxmin()}' has <2 rows; "
                "cannot cross-validate. Adjust variables or data."}

    def _new_model():
        return LogisticRegression(max_iter=2000, C=C)

    # ---- point estimate: fit on all complete rows --------------------------
    clf = _new_model().fit(X, y)
    coef = clf.coef_                       # (n_eff, n_features)
    coef_classes = (clf.classes_ if coef.shape[0] == len(clf.classes_)
                    else [f"{clf.classes_[1]} vs {clf.classes_[0]}"])

    # ---- out-of-fold predictions for honest behavior views -----------------
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(_new_model(), X, y, cv=skf, method="predict")
    proba = cross_val_predict(_new_model(), X, y, cv=skf, method="predict_proba")
    proba_classes = np.unique(y)           # cross_val_predict proba col order
    max_prob = proba.max(axis=1)

    labels = list(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels)

    # ---- point metrics -----------------------------------------------------
    acc = accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred, labels=labels, average=None, zero_division=0)

    # ---- bootstrap CI on metrics (resample the OOF (y, y_pred) pairs) -------
    boot_acc, boot_rec = [], {l: [] for l in labels}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y[idx], y_pred[idx]
        boot_acc.append(accuracy_score(yt, yp))
        for l in labels:
            m = yt == l
            boot_rec[l].append((yp[m] == l).mean() if m.any() else np.nan)

    # ---- bootstrap CI on coefficients (resample rows, refit) ---------------
    coef_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < len(classes):
            continue
        try:
            mb = _new_model().fit(X[idx], yb)
        except Exception:
            continue
        if not np.array_equal(mb.classes_, clf.classes_):
            continue
        if mb.coef_.shape != coef.shape:
            continue
        coef_boot.append(mb.coef_)
    coef_boot = np.array(coef_boot) if coef_boot else np.empty((0,) + coef.shape)

    # ---- assemble parameter table ------------------------------------------
    param_rows = []
    for ci in range(coef.shape[0]):
        cls_label = (coef_classes[ci] if isinstance(coef_classes, (list, np.ndarray))
                     and len(np.atleast_1d(coef_classes)) > ci
                     else str(coef_classes))
        for fi, fname in enumerate(feat_names):
            if coef_boot.shape[0] > 0:
                col = coef_boot[:, ci, fi]
                var = float(np.nanvar(col, ddof=1))
                lo, hi = _percentile_ci(col)
            else:
                var, lo, hi = np.nan, np.nan, np.nan
            param_rows.append({
                "class": str(cls_label),
                "variable": fname,
                "point_estimate": float(coef[ci, fi]),
                "variance": var,
                "std_error": float(np.sqrt(var)) if var == var else np.nan,
                "ci_low": lo,
                "ci_high": hi,
            })
    parameters = pd.DataFrame(param_rows)

    # ---- metrics table ------------------------------------------------------
    metric_rows = [{
        "metric": "accuracy", "scope": "overall", "value": float(acc),
        "ci_low": _percentile_ci(boot_acc)[0], "ci_high": _percentile_ci(boot_acc)[1],
    }]
    for l, r in zip(labels, rec):
        lo, hi = _percentile_ci(boot_rec[l])
        metric_rows.append({"metric": "sensitivity (recall)", "scope": str(l),
                            "value": float(r), "ci_low": lo, "ci_high": hi})
    metrics = pd.DataFrame(metric_rows)

    # ---- predictions + failure cases ---------------------------------------
    pred = meta.copy()
    pred["actual_diagnosis"] = y
    pred["predicted_diagnosis"] = y_pred
    pred["max_prob"] = np.round(max_prob, 3)
    for j, c in enumerate(proba_classes):
        pred[f"prob_{c}"] = np.round(proba[:, j], 3)

    failures = pred[pred["actual_diagnosis"] != pred["predicted_diagnosis"]][
        ["subject_id", "session", "actual_diagnosis", "predicted_diagnosis",
         "max_prob"]].reset_index(drop=True)

    return {
        "ok": True,
        "task": "classification",
        "meta": {
            "model": "LogisticRegression",
            "target": target_col,
            "variables": list(selected_vars),
            "encoded_features": feat_names,
            "n_rows_used": int(n),
            "n_rows_dropped": int(dropped),
            "cv_folds": int(folds),
            "n_boot": int(n_boot),
            "impute": bool(impute),
            "classes": [str(c) for c in labels],
        },
        "parameters": parameters,
        "metrics": metrics,
        "confusion_matrix": {"labels": [str(l) for l in labels],
                             "matrix": cm.tolist()},
        "predictions": pred,
        "failure_cases": failures,
    }


# --------------------------------------------------------------------------- #
# REGRESSION (extra model to show the contract is model-agnostic)
# --------------------------------------------------------------------------- #
def fit_regression(feature_df, selected_vars, target_col="behavioral_score",
                   n_boot=200, n_folds=5, seed=0, impute=False, alpha=1.0):
    rng = np.random.default_rng(seed)
    sel = [v for v in selected_vars if v != target_col]
    X, y, meta, feat_names, dropped = _assemble(feature_df, sel, target_col, impute)
    n = len(y) if y is not None else 0
    if n < 8 or X.shape[1] == 0:
        return {"ok": False, "error": f"Too few complete rows (n={n})."}

    folds = max(2, min(n_folds, n))
    model = Ridge(alpha=alpha).fit(X, y)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(Ridge(alpha=alpha), X, y, cv=kf)

    coef_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        coef_boot.append(Ridge(alpha=alpha).fit(X[idx], y[idx]).coef_)
    coef_boot = np.array(coef_boot)

    params = pd.DataFrame([{
        "variable": f, "point_estimate": float(model.coef_[i]),
        "variance": float(np.nanvar(coef_boot[:, i], ddof=1)),
        "ci_low": _percentile_ci(coef_boot[:, i])[0],
        "ci_high": _percentile_ci(coef_boot[:, i])[1],
    } for i, f in enumerate(feat_names)])

    metrics = pd.DataFrame([
        {"metric": "R2", "value": float(r2_score(y, y_pred))},
        {"metric": "MAE", "value": float(mean_absolute_error(y, y_pred))},
    ])
    pred = meta.copy()
    pred["actual"] = y
    pred["predicted"] = np.round(y_pred, 3)
    return {"ok": True, "task": "regression",
            "meta": {"model": "Ridge", "target": target_col,
                     "variables": sel, "n_rows_used": int(n),
                     "n_rows_dropped": int(dropped)},
            "parameters": params, "metrics": metrics, "predictions": pred}


# --------------------------------------------------------------------------- #
# CLUSTERING (unsupervised; no labels needed)
# --------------------------------------------------------------------------- #
def fit_clustering(feature_df, selected_vars, k=3, seed=0, impute=False):
    X, _, meta, feat_names, dropped = _assemble(feature_df, selected_vars,
                                                None, impute)
    n = X.shape[0]
    if n < k or X.shape[1] == 0:
        return {"ok": False, "error": f"Too few complete rows (n={n})."}
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    sil = float(silhouette_score(X, km.labels_)) if k < n else np.nan
    meta = meta.copy()
    meta["cluster"] = km.labels_
    crosstab = None
    if "true_diagnosis" in meta.columns:
        crosstab = pd.crosstab(meta["cluster"], meta["true_diagnosis"])
    return {"ok": True, "task": "clustering",
            "meta": {"model": "KMeans", "k": k, "variables": list(selected_vars),
                     "n_rows_used": int(n), "n_rows_dropped": int(dropped),
                     "silhouette": sil},
            "assignments": meta, "crosstab": crosstab}


# --------------------------------------------------------------------------- #
# Serialization for the unified results artifact
# --------------------------------------------------------------------------- #
def results_to_json_bytes(results: dict) -> bytes:
    """JSON-serializable view of a classification results dict."""
    out = {
        "ok": results.get("ok"),
        "task": results.get("task"),
        "meta": results.get("meta"),
        "confusion_matrix": results.get("confusion_matrix"),
        "parameters": results["parameters"].to_dict("records")
        if isinstance(results.get("parameters"), pd.DataFrame) else None,
        "metrics": results["metrics"].to_dict("records")
        if isinstance(results.get("metrics"), pd.DataFrame) else None,
        "failure_cases": results["failure_cases"].to_dict("records")
        if isinstance(results.get("failure_cases"), pd.DataFrame) else None,
    }
    return json.dumps(out, indent=2, default=str).encode("utf-8")
