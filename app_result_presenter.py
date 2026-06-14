"""
app_result_presenter.py
=======================
APP 3 of 3 - the "result presenter".

Built on top of the dataset generator and the model-fit generator. Layout
follows the requested three-part structure:

  Part 1  Variable selection (st.multiselect) -> feeds the model-fit module live
  Part 2  Missingness heatmap + descriptive statistics of the chosen variables
  Part 3  Results: parameter table (point estimate / variance / CI), then model
          behavior - confusion matrix, per-class sensitivity, failure-case list

The fit itself is delegated to model_fit.py (the same module the model-fit app
uses), so this app stays a presentation layer.

Deploy: this file + synthetic_data.py, model_fit.py, app_common.py,
requirements.txt; point Streamlit Cloud here.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import model_fit as mf
from app_common import load_dataset_ui

st.set_page_config(page_title="Result Presenter", layout="wide")
st.title("Result Presenter")
st.caption("Select variables, inspect data quality, and read model behavior "
           "through several lenses - each with an uncertainty band, since there "
           "is no single agreed metric.")

clinical, embeddings, ground_truth = load_dataset_ui("pres")
if clinical is None:
    st.stop()

feature_df = mf.build_feature_table(clinical, embeddings, ground_truth)
candidates = mf.available_candidates(feature_df)

# --------------------------------------------------------------------------- #
# PART 1 - variable selection
# --------------------------------------------------------------------------- #
st.divider()
st.header("Part 1 - Choose variables")
default = [v for v in ["emb_pc1", "emb_pc2", "emb_pc3", "age", "quality_score"]
           if v in candidates]
selected = st.multiselect("Variables to keep in the model", candidates,
                          default=default)

c1, c2, c3, c4 = st.columns(4)
label_source = c1.selectbox("Diagnosis label",
                            ["observed (clinical)", "oracle (ground truth)"])
n_boot = c2.slider("Bootstrap resamples", 50, 500, 200, 50)
n_folds = c3.slider("CV folds", 2, 10, 5)
impute = c4.checkbox("Impute missing", value=False)
target_col = "diagnosis" if label_source.startswith("observed") else "true_diagnosis"

run = st.button("Run / refresh results", type="primary")
if run:
    st.session_state["pres_res"] = mf.fit_and_report(
        feature_df, selected, target_col=target_col, n_boot=n_boot,
        n_folds=n_folds, seed=0, impute=impute)

# --------------------------------------------------------------------------- #
# PART 2 - data quality
# --------------------------------------------------------------------------- #
st.divider()
st.header("Part 2 - Data quality")

if not selected:
    st.info("Select at least one variable above.")
else:
    cols_for_view = selected + [target_col]
    sub = feature_df[["subject_id", "session"] + cols_for_view].copy()

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Missingness heatmap")
        miss = sub[cols_for_view].isna().T.astype(int)  # vars x rows
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(cols_for_view) + 1.2))
        ax.imshow(miss.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_yticks(range(len(cols_for_view)), cols_for_view)
        ax.set_xlabel(f"{miss.shape[1]} subject-sessions "
                      "(red = missing, white = present)")
        ax.set_xticks([])
        st.pyplot(fig)
        pct = (sub[cols_for_view].isna().mean() * 100).round(1)
        st.caption("Percent missing per variable: " +
                   ", ".join(f"{k} {v}%" for k, v in pct.items()))

    with right:
        st.subheader("Descriptive statistics")
        num = [v for v in selected if v in mf.CONTINUOUS_VARS]
        if num:
            st.dataframe(sub[num].describe().round(3), use_container_width=True)
        cats = [v for v in selected if v in mf.CATEGORICAL_VARS] + [target_col]
        for cvar in cats:
            if cvar in sub.columns:
                st.write(f"**{cvar}**")
                st.dataframe(sub[cvar].value_counts(dropna=False)
                             .rename_axis(cvar).reset_index(name="count"),
                             use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# PART 3 - results
# --------------------------------------------------------------------------- #
st.divider()
st.header("Part 3 - Results")

res = st.session_state.get("pres_res")
if not res:
    st.info("Press **Run / refresh results** above to fit and display.")
elif not res.get("ok"):
    st.error(res["error"])
else:
    m = res["meta"]
    st.caption(f"Model: {m['model']} | target: {m['target']} | rows used: "
               f"{m['n_rows_used']} (dropped {m['n_rows_dropped']} for "
               f"missingness) | CV folds: {m['cv_folds']} | bootstrap: "
               f"{m['n_boot']}")

    # ---- parameter table ---------------------------------------------------
    st.subheader("Parameters")
    st.caption("Point estimate with bootstrap variance and 95% confidence "
               "interval.")
    st.dataframe(res["parameters"].round(4), use_container_width=True,
                 hide_index=True)

    # coefficient plot with CI
    p = res["parameters"].copy()
    p["label"] = p["class"] + " : " + p["variable"]
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(p) + 1))
    yloc = np.arange(len(p))
    ax.errorbar(p["point_estimate"], yloc,
                xerr=[p["point_estimate"] - p["ci_low"],
                      p["ci_high"] - p["point_estimate"]],
                fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(yloc, p["label"], fontsize=8)
    ax.set_xlabel("coefficient (standardized)")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.divider()
    st.subheader("Model behavior")

    bc1, bc2 = st.columns([1, 1])
    with bc1:
        st.markdown("**Confusion matrix** (out-of-fold)")
        cm = np.array(res["confusion_matrix"]["matrix"])
        labels = res["confusion_matrix"]["labels"]
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.8))
        ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        ax2.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax2.text(j, i, cm[i, j], ha="center", va="center")
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
        st.pyplot(fig2)

    with bc2:
        st.markdown("**Sensitivity (recall) by class**")
        sens = res["metrics"][res["metrics"]["metric"]
                              .str.startswith("sensitivity")].copy()
        sens = sens.rename(columns={"scope": "class"})[
            ["class", "value", "ci_low", "ci_high"]]
        st.dataframe(sens.round(3), use_container_width=True, hide_index=True)
        acc = res["metrics"].iloc[0]
        st.metric("Overall accuracy",
                  f"{acc['value']:.2f}",
                  help=f"95% CI [{acc['ci_low']:.2f}, {acc['ci_high']:.2f}]")
        st.caption(f"95% CI [{acc['ci_low']:.2f}, {acc['ci_high']:.2f}]")

    st.markdown(f"**Failure cases** ({len(res['failure_cases'])} of "
                f"{m['n_rows_used']})")
    st.caption("Subjects the model predicted incorrectly out-of-fold.")
    st.dataframe(res["failure_cases"], use_container_width=True, hide_index=True)
