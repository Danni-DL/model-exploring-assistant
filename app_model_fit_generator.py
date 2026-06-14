"""
app_model_fit_generator.py
==========================
APP 2 of 3 - the "model fit generator".

Purpose: take a dataset, fit a deliberately simple demo model, and emit the
unified RESULTS CONTRACT (results.json + CSVs) that the presenter consumes.

The primary model is logistic regression on selected variables -> diagnosis.
Two extra models (ridge regression, KMeans clustering) are included on separate
tabs to demonstrate the contract is model-agnostic: the research group has not
chosen a model, so the tool must not depend on one.

Deploy: put this + synthetic_data.py, model_fit.py, app_common.py,
requirements.txt in a repo and point Streamlit Cloud at this file.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import model_fit as mf
from app_common import load_dataset_ui

st.set_page_config(page_title="Model Fit Generator", layout="wide")
st.title("Model Fit Generator")
st.caption("Fits a simple demo model and exports a model-agnostic results "
           "artifact. The tool defines the OUTPUT format; the model is "
           "swappable.")

clinical, embeddings, ground_truth = load_dataset_ui("fit")
if clinical is None:
    st.stop()

feature_df = mf.build_feature_table(clinical, embeddings, ground_truth)
candidates = mf.available_candidates(feature_df)

st.divider()
st.subheader("2. Fit settings")
s1, s2, s3, s4 = st.columns(4)
label_source = s1.selectbox("Diagnosis label source",
                            ["observed (clinical)", "oracle (ground truth)"],
                            help="Observed labels are missing at some sites "
                                 "(realistic). Oracle uses the synthetic truth.")
n_boot = s2.slider("Bootstrap resamples", 50, 500, 200, 50)
n_folds = s3.slider("CV folds", 2, 10, 5)
seed = s4.number_input("Seed", 0, 9999, 0, 1)
impute = st.checkbox("Mean-impute missing features (else complete-case rows)",
                     value=False)
target_col = "diagnosis" if label_source.startswith("observed") else "true_diagnosis"

tab_clf, tab_reg, tab_clu = st.tabs(
    ["Classification (primary)", "Regression", "Clustering"])

# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
with tab_clf:
    default = [v for v in ["emb_pc1", "emb_pc2", "emb_pc3", "age",
                           "quality_score"] if v in candidates]
    sel = st.multiselect("Predictor variables", candidates, default=default,
                         key="clf_vars")
    if st.button("Fit logistic regression", type="primary"):
        res = mf.fit_and_report(feature_df, sel, target_col=target_col,
                                n_boot=n_boot, n_folds=n_folds, seed=seed,
                                impute=impute)
        if not res["ok"]:
            st.error(res["error"])
        else:
            st.session_state["clf_res"] = res

    res = st.session_state.get("clf_res")
    if res and res.get("ok"):
        m = res["meta"]
        a, b, c, d = st.columns(4)
        a.metric("Rows used", m["n_rows_used"])
        b.metric("Rows dropped (missing)", m["n_rows_dropped"])
        c.metric("CV folds", m["cv_folds"])
        d.metric("Classes", len(m["classes"]))

        st.markdown("**Parameter table** (point estimate, bootstrap variance, "
                    "95% CI)")
        st.dataframe(res["parameters"].round(4), use_container_width=True,
                     hide_index=True)

        st.markdown("**Metrics** (with bootstrap 95% CI)")
        st.dataframe(res["metrics"].round(4), use_container_width=True,
                     hide_index=True)

        st.markdown("**Confusion matrix** (out-of-fold)")
        cm = np.array(res["confusion_matrix"]["matrix"])
        labels = res["confusion_matrix"]["labels"]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown(f"**Failure cases** ({len(res['failure_cases'])})")
        st.dataframe(res["failure_cases"], use_container_width=True,
                     hide_index=True)

        st.divider()
        st.subheader("3. Export results artifact")
        e1, e2, e3 = st.columns(3)
        e1.download_button("results.json",
                           mf.results_to_json_bytes(res), "results.json",
                           "application/json", use_container_width=True)
        e2.download_button("parameters.csv",
                           res["parameters"].to_csv(index=False).encode(),
                           "parameters.csv", "text/csv", use_container_width=True)
        e3.download_button("predictions.csv",
                           res["predictions"].to_csv(index=False).encode(),
                           "predictions.csv", "text/csv",
                           use_container_width=True)

# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
with tab_reg:
    st.caption("Predicts the continuous behavioral score - same contract, "
               "different model type.")
    rdefault = [v for v in ["emb_pc1", "emb_pc2", "emb_pc3", "age"]
                if v in candidates]
    rsel = st.multiselect("Predictor variables", candidates, default=rdefault,
                          key="reg_vars")
    if st.button("Fit ridge regression"):
        rr = mf.fit_regression(feature_df, rsel, n_boot=n_boot,
                               n_folds=n_folds, seed=seed, impute=impute)
        if not rr["ok"]:
            st.error(rr["error"])
        else:
            st.dataframe(rr["metrics"].round(4), hide_index=True)
            st.dataframe(rr["parameters"].round(4), use_container_width=True,
                         hide_index=True)

# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
with tab_clu:
    st.caption("Unsupervised - no labels needed. Compares discovered clusters "
               "to the (synthetic) true diagnosis.")
    cdefault = [v for v in ["emb_pc1", "emb_pc2", "emb_pc3"] if v in candidates]
    csel = st.multiselect("Variables", candidates, default=cdefault,
                          key="clu_vars")
    k = st.slider("Clusters (k)", 2, 6, 3)
    if st.button("Fit KMeans"):
        cc = mf.fit_clustering(feature_df, csel, k=k, seed=seed, impute=impute)
        if not cc["ok"]:
            st.error(cc["error"])
        else:
            st.metric("Silhouette", round(cc["meta"]["silhouette"], 3))
            if cc["crosstab"] is not None:
                st.markdown("**Cluster vs. true diagnosis**")
                st.dataframe(cc["crosstab"])
