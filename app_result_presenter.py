"""
app_result_presenter.py
=======================
APP 3 of 3 - the "result presenter".

Built on top of the dataset generator and the model-fit generator. Layout
follows the requested three-part structure:

  1. Data source  Embedded dataset generator - tune Generation parameters, the
                  evolving model, and Difficulty; data regenerates live.
  Part 1  Variable selection (st.multiselect) -> feeds the model-fit module live
  Part 2  Missingness heatmap + descriptive statistics of the chosen variables
  Part 3  Results: parameter table (point estimate / variance / CI), then model
          behavior - confusion matrix, per-class sensitivity, failure-case list.

Plain-language interpretations
------------------------------
Each result block in Part 3 carries a small "Explain this" button. The page
shows the raw data/figure by default; clicking the button opens a pop-over
card (st.popover) with a detailed, non-technical interpretation. The wording is
templated and fixed; only the numbers, class names, and directions change with
each run, so a non-statistical reader sees a consistent explanation that still
reflects the current result.

The fit itself is delegated to model_fit.py (the same module the model-fit app
uses), so this app stays a presentation layer.

Deploy: this file + synthetic_data.py, model_fit.py, requirements.txt; point
Streamlit Cloud here.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import synthetic_data as sd
import model_fit as mf

st.set_page_config(page_title="Result Presenter", layout="wide")
st.title("Result Presenter")
st.caption("Generate data, select variables, inspect data quality, and read "
           "model behavior through several lenses - each with an uncertainty "
           "band, since there is no single agreed metric. Click any "
           "\u201cExplain this\u201d button for a plain-language walkthrough.")


# --------------------------------------------------------------------------- #
# Plain-language interpretation templates (fixed wording, numbers fill in)
# --------------------------------------------------------------------------- #
def _ci_excludes_zero(lo, hi) -> bool:
    if pd.isna(lo) or pd.isna(hi):
        return False
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def interpret_parameters(params: pd.DataFrame, small_data: bool) -> str:
    n = len(params)
    clear = params[params.apply(
        lambda r: _ci_excludes_zero(r["ci_low"], r["ci_high"]), axis=1)]
    k = len(clear)

    out = []
    out.append(
        "**What this table is.** The model weighs several measurements to guess "
        "each diagnosis. Every row is one measurement's relationship to one "
        "diagnosis.")
    out.append(
        "**What each column means.**\n"
        "- *variable* - the input being used. For example `emb_pc1` is the "
        "first principal component of the high-dimensional embedding (the "
        "direction along which the model's features vary the most); `age` and "
        "`quality_score` are what they sound like. Continuous variables are "
        "standardized, so the sizes of different coefficients are comparable.\n"
        "- *point_estimate* - the direction and strength of the relationship. "
        "Positive means a higher value pushes *toward* that diagnosis; negative "
        "means it pushes *away*. Bigger size = stronger effect.\n"
        "- *variance / std_error* - how much that estimate wobbles. We resample "
        "the data and refit many times (bootstrap) and watch how much the "
        "coefficient moves; more movement = less certainty. Std error is the "
        "square root of the variance.\n"
        "- *ci_low / ci_high* - the 95% confidence interval: the range the true "
        "coefficient most likely falls in (the middle 95% of those bootstrap "
        "refits).")
    out.append(
        "**Does the interval include 0?** If the whole interval sits on one "
        "side of zero (e.g. 0.4 to 1.2), the effect is *statistically "
        "significant* - we can trust it is real. If the interval crosses zero "
        "(e.g. -0.3 to 0.6), it is *not significant*.")
    out.append(
        "**\u201cNot significant\u201d in plain words:** it does **not** mean "
        "the variable is useless. It means *we do not have enough data to tell "
        "whether it matters* - the small effect we see could just be random "
        "noise. Not significant = \u201cnot enough evidence,\u201d not "
        "\u201cproven irrelevant.\u201d This happens easily when the dataset is "
        "small.")
    out.append(
        "**How to read the figure.** Each dot is a coefficient's point "
        "estimate; the horizontal line is its confidence interval; the dashed "
        "vertical line is zero. A line that touches or crosses zero = not "
        "significant. A longer line = more uncertainty (usually too few "
        "subjects).")
    if k == 0:
        out.append(
            f"**This run:** none of the {n} relationships are statistically "
            f"clear (every interval includes zero). That usually means the "
            f"dataset is too small or the chosen variables do not separate the "
            f"groups well.")
    else:
        top = clear.reindex(
            clear["point_estimate"].abs().sort_values(ascending=False).index
        ).iloc[0]
        direction = "positive" if top["point_estimate"] > 0 else "negative"
        hl = "higher" if top["point_estimate"] > 0 else "lower"
        out.append(
            f"**This run:** {k} of {n} relationships are reliable (interval "
            f"excludes zero). The clearest: for '{top['class']}', "
            f"'{top['variable']}' has a {direction} effect (about "
            f"{top['point_estimate']:.2f}) - a {hl} value makes "
            f"'{top['class']}' more likely.")
    if small_data:
        out.append(
            "Several intervals are wide because the dataset is small, so treat "
            "weak signals cautiously.")
    return "\n\n".join(out)


def interpret_confusion(cm, labels) -> str:
    cm = np.array(cm)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    out = []
    out.append(
        "**What this grid is.** Rows are the *true* diagnosis; columns are the "
        "model's *guess*. Cells on the diagonal (top-left to bottom-right) are "
        "correct; every cell off the diagonal is a mistake.")
    out.append(
        f"**This run:** out of {total} cases the model got {correct} right "
        f"({correct / total:.0%})." if total else "No cases to summarize.")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.sum() == 0:
        out.append("It made no mistakes in this run.")
    else:
        i, j = np.unravel_index(np.argmax(off), off.shape)
        out.append(
            f"Its most common mistake was labeling true '{labels[i]}' cases as "
            f"'{labels[j]}' ({int(off[i, j])} times).")
        perfect = [labels[r] for r in range(len(labels))
                   if off[r].sum() == 0 and cm[r].sum() > 0]
        if perfect:
            out.append(f"It identified '{perfect[0]}' without error.")
    out.append(
        "**Why it beats a single accuracy number:** it shows *which* groups get "
        "confused with *which*, not just how often the model is wrong.")
    return "\n\n".join(out)


def interpret_sensitivity(sens: pd.DataFrame) -> str:
    out = []
    out.append(
        "**What sensitivity (recall) means.** Of all the people who *truly* "
        "have a given diagnosis, what fraction did the model correctly catch? "
        "100% means it missed no one; 50% means it missed half. It is different "
        "from accuracy: accuracy is overall correctness, sensitivity is about "
        "*not missing* one specific group.")
    s = sens.dropna(subset=["value"])
    if len(s):
        best = s.loc[s["value"].idxmax()]
        worst = s.loc[s["value"].idxmin()]
        out.append(
            f"**This run:** the model is best at catching '{best['class']}' "
            f"({best['value']:.0%}) and weakest at '{worst['class']}' "
            f"({worst['value']:.0%}).")
        if worst["value"] < 0.7:
            out.append(
                f"A low value for '{worst['class']}' means many real cases were "
                f"labeled as something else - which matters a lot if missing "
                f"them is costly.")
    out.append(
        "The bracketed range after each number is its confidence interval - how "
        "much the value could shift given the limited data.")
    return "\n\n".join(out)


def interpret_failures(failures: pd.DataFrame, n_used: int) -> str:
    k = len(failures)
    if k == 0:
        return ("**No mistakes on held-out subjects this run.** With a small "
                "dataset, do not read this as proof the model is perfect - it "
                "may simply not have been tested hard enough.")
    rate = (k / n_used) if n_used else float("nan")
    out = []
    out.append(
        f"**What this list is.** Each row is a specific subject the model got "
        f"wrong *when it had not seen them during training* (out-of-fold), so "
        f"these are honest mistakes. There are {k} of them out of {n_used} "
        f"({rate:.0%}).")
    out.append(
        "**Columns:** `actual_diagnosis` is the truth; `predicted_diagnosis` is "
        "the model's guess; `max_prob` is how confident the model was, from 0 "
        "to 1.")
    if "max_prob" in failures.columns and failures["max_prob"].notna().any():
        avg = failures["max_prob"].mean()
        hi = int((failures["max_prob"] > 0.7).sum())
        if hi > 0:
            out.append(
                f"{hi} of these mistakes were made with high confidence (above "
                f"0.70) - the model was sure but wrong, so these subjects are "
                f"worth inspecting first.")
        else:
            out.append(
                f"Most mistakes came with low confidence (average {avg:.2f}), "
                f"suggesting the model was already unsure about them.")
    out.append(
        "**Why it is here:** it is a clickable entry point for clinical or "
        "research staff to look at the actual subjects behind the errors - more "
        "concrete than any summary metric.")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# 1. DATA SOURCE - embedded dataset generator
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Generating data...")
def _generate(kwargs: dict):
    cfg = sd.GenConfig(**kwargs)
    d = sd.generate(cfg)
    return d["clinical"], d["embeddings"], d["ground_truth"]


st.subheader("1. Data source")
st.caption("Dataset generator (embedded). Adjust the parameters below and the "
           "dataset is regenerated live - no upload needed.")

st.markdown("**Generation parameters**")
g1, g2, g3, g4, g5 = st.columns(5)
n_subjects = g1.slider("Subjects", 10, 200, 60, 5,
                       help="More subjects -> more rows to fit and tighter CIs.")
n_sessions = g2.slider("Sessions per subject", 1, 5, 2)
n_sites = g3.slider("Sites", 1, 5, 3)
n_classes = g4.slider("Diagnosis classes", 2, 5, 3)
embedding_dim = g5.select_slider("Embedding dim",
                                 options=[16, 32, 64, 128, 256], value=128)

st.markdown("**The 'evolving model'**")
model_version = st.slider("Model version", 1, 6, 1,
                          help="Reseeds the embedding projection + noise, so the "
                               "same subject yields different embeddings across "
                               "versions.")

st.markdown("**Difficulty**")
d1, d2, d3 = st.columns(3)
missingness = d1.slider("Missingness intensity", 0.0, 2.0, 1.0, 0.1,
                        help="Global multiplier on dropout, modality gaps, and "
                             "clinical missingness.")
progression = d2.slider("Longitudinal progression", 0.0, 1.5, 0.6, 0.1)
seed = d3.number_input("Random seed", 0, 10_000, 7, 1)

clinical, embeddings, ground_truth = _generate(dict(
    n_subjects=n_subjects, n_sessions=n_sessions, n_sites=n_sites,
    n_classes=n_classes, embedding_dim=embedding_dim,
    model_version=model_version, missingness=missingness,
    progression=progression, seed=seed,
))
st.caption(f"Generated {clinical.shape[0]} session rows across "
           f"{ground_truth.shape[0]} subjects (model version {model_version}).")

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
    small_data = m["n_rows_used"] < 80
    st.caption(f"Model: {m['model']} | target: {m['target']} | rows used: "
               f"{m['n_rows_used']} (dropped {m['n_rows_dropped']} for "
               f"missingness) | CV folds: {m['cv_folds']} | bootstrap: "
               f"{m['n_boot']}")

    # ---- parameter table ---------------------------------------------------
    ph1, ph2 = st.columns([3, 1])
    ph1.subheader("Parameters")
    with ph2.popover("💬 Explain this"):
        st.markdown(interpret_parameters(res["parameters"], small_data))
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
        h1, h2 = st.columns([3, 1])
        h1.markdown("**Confusion matrix** (out-of-fold)")
        with h2.popover("💬 Explain this"):
            st.markdown(interpret_confusion(res["confusion_matrix"]["matrix"],
                                            res["confusion_matrix"]["labels"]))
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
        h3, h4 = st.columns([3, 1])
        h3.markdown("**Sensitivity (recall) by class**")
        sens = res["metrics"][res["metrics"]["metric"]
                              .str.startswith("sensitivity")].copy()
        sens = sens.rename(columns={"scope": "class"})[
            ["class", "value", "ci_low", "ci_high"]]
        with h4.popover("💬 Explain this"):
            st.markdown(interpret_sensitivity(sens))
        st.dataframe(sens.round(3), use_container_width=True, hide_index=True)
        acc = res["metrics"].iloc[0]
        st.metric("Overall accuracy", f"{acc['value']:.2f}",
                  help=f"95% CI [{acc['ci_low']:.2f}, {acc['ci_high']:.2f}]")
        st.caption(f"95% CI [{acc['ci_low']:.2f}, {acc['ci_high']:.2f}]")

    fh1, fh2 = st.columns([3, 1])
    fh1.markdown(f"**Failure cases** ({len(res['failure_cases'])} of "
                 f"{m['n_rows_used']})")
    with fh2.popover("💬 Explain this"):
        st.markdown(interpret_failures(res["failure_cases"], m["n_rows_used"]))
    st.caption("Subjects the model predicted incorrectly out-of-fold.")
    st.dataframe(res["failure_cases"], use_container_width=True, hide_index=True)
