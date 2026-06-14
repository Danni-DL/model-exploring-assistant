"""
app.py
======
Streamlit Cloud app that generates and previews the synthetic multimodal
biomedical dataset for the CSP assessment, and lets you download every artifact
(individually or as a single ZIP bundle).

Deploy on Streamlit Community Cloud:
  1. Put app.py, synthetic_data.py, requirements.txt in a public GitHub repo.
  2. share.streamlit.io -> New app -> point at app.py.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py
"""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

import synthetic_data as sd


st.set_page_config(page_title="Synthetic Multimodal Biomedical Data",
                   layout="wide")

st.title("Synthetic Multimodal Biomedical Dataset Generator")
st.caption(
    "Generates data matching the assessment scenario: an evolving model on "
    "high-dimensional biomedical data, multimodal (embeddings + sensors + "
    "tabular), longitudinal, heterogeneous across sites, with meaningful "
    "missingness and a known ground truth for validation."
)

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Generation parameters")

    n_subjects = st.slider("Subjects", 10, 200, 30, 5)
    n_sessions = st.slider("Sessions per subject", 1, 5, 2)
    n_sites = st.slider("Sites", 1, 5, 3)
    n_classes = st.slider("Diagnosis classes", 2, 5, 3)
    embedding_dim = st.select_slider("Embedding dim",
                                     options=[16, 32, 64, 128, 256], value=128)

    st.divider()
    st.subheader("The 'evolving model'")
    model_version = st.slider("Model version", 1, 6, 1,
                              help="Reseeds the embedding projection + noise, so "
                                   "the same subject yields different embeddings "
                                   "across versions.")

    st.divider()
    st.subheader("Difficulty")
    missingness = st.slider("Missingness intensity", 0.0, 2.0, 1.0, 0.1,
                            help="Global multiplier on dropout, modality gaps, "
                                 "and clinical missingness.")
    progression = st.slider("Longitudinal progression", 0.0, 1.5, 0.6, 0.1)
    seed = st.number_input("Random seed", 0, 10_000, 7, 1)

    st.divider()
    generate_clicked = st.button("Generate dataset", type="primary",
                                 use_container_width=True)


# --------------------------------------------------------------------------- #
# Generate (cached on the full config so re-runs are cheap)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Generating synthetic data...")
def _gen(cfg_kwargs: dict):
    cfg = sd.GenConfig(**cfg_kwargs)
    return sd.generate(cfg)


cfg_kwargs = dict(
    n_subjects=n_subjects, n_sessions=n_sessions, n_sites=n_sites,
    n_classes=n_classes, embedding_dim=embedding_dim,
    model_version=model_version, missingness=missingness,
    progression=progression, seed=seed,
)

# generate on first load and whenever params change
data = _gen(cfg_kwargs)
clinical = data["clinical"]
embeddings = data["embeddings"]
sensors = data["sensors"]
ground_truth = data["ground_truth"]

# --------------------------------------------------------------------------- #
# Top-line summary
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Subjects", ground_truth["subject_id"].nunique())
c2.metric("Sessions present", len(clinical))
expected = n_subjects * n_sessions
c3.metric("Session dropout",
          f"{100 * (1 - len(clinical) / expected):.0f}%")
c4.metric("Embedding rows", len(embeddings))
c5.metric("Sensor samples", f"{len(sensors):,}")

st.divider()

left, right = st.columns([1, 1])

# --------------------------------------------------------------------------- #
# Cross-site completeness (the "data varies across sites" picture)
# --------------------------------------------------------------------------- #
with left:
    st.subheader("Cross-site completeness")
    st.caption("Each site differs in label schema, modality coverage, quality, "
               "and dropout. This is the heterogeneity a tool has to absorb.")
    summary = sd.missingness_summary(clinical, n_subjects, n_sessions)
    st.dataframe(summary, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Embedding structure (PCA), colored by ground-truth class
# --------------------------------------------------------------------------- #
with right:
    st.subheader("Embedding space (PCA)")
    st.caption("2D PCA of the high-dim embeddings, colored by the TRUE class. "
               "Recoverable structure here is what makes synthetic-data "
               "validation possible.")
    if len(embeddings) >= 2:
        meta, X = sd.embedding_matrix(embeddings)
        coords = sd.pca_2d(X)
        plot_df = meta.copy()
        plot_df["PC1"] = coords[:, 0]
        plot_df["PC2"] = coords[:, 1]
        plot_df = plot_df.merge(
            ground_truth[["subject_id", "true_diagnosis"]],
            on="subject_id", how="left")
        st.scatter_chart(plot_df, x="PC1", y="PC2", color="true_diagnosis",
                         height=360)
    else:
        st.info("Not enough embedding rows to plot.")

st.divider()

# --------------------------------------------------------------------------- #
# A sample sensor trace
# --------------------------------------------------------------------------- #
st.subheader("Example sensor time series")
if len(sensors) > 0:
    sample_ids = sensors["subject_id"].unique()
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        sid = st.selectbox("Subject", sample_ids)
        avail_sess = sorted(sensors.loc[sensors.subject_id == sid, "session"].unique())
        sess = st.selectbox("Session", avail_sess)
    sub = sensors[(sensors.subject_id == sid) & (sensors.session == sess)]
    wide = sub.pivot_table(index="t", columns="channel", values="value")
    wide.columns = [f"channel_{c}" for c in wide.columns]
    with sc2:
        st.line_chart(wide, height=260)
    st.caption(f"Sampling rate at this site: {int(sub['fs'].iloc[0])} Hz. "
               "Sampling rates differ across sites, so series lengths differ.")
else:
    st.info("No sensor data in this configuration (all sensor-collecting sites "
            "may be off).")

st.divider()

# --------------------------------------------------------------------------- #
# Data previews
# --------------------------------------------------------------------------- #
with st.expander("Preview tables"):
    st.write("**Clinical / tabular (one row per existing subject-session)**")
    st.dataframe(clinical.head(20), use_container_width=True, hide_index=True)
    st.write("**Embeddings (truncated columns)**")
    show_cols = (["subject_id", "session", "site", "model_version"] +
                 [c for c in embeddings.columns if c.startswith("emb_")][:6])
    st.dataframe(embeddings[show_cols].head(20) if len(embeddings) else embeddings,
                 use_container_width=True, hide_index=True)
    st.write("**Ground truth (oracle — not available on real data)**")
    st.dataframe(ground_truth.head(20), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _bundle_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("clinical.csv", _csv(clinical))
        z.writestr("embeddings.csv", _csv(embeddings))
        z.writestr("sensors.csv", _csv(sensors))
        z.writestr("ground_truth.csv", _csv(ground_truth))
        z.writestr("config.json", json.dumps(data["config"], indent=2))
        # compact array form for ML consumers
        if len(embeddings):
            meta, X = sd.embedding_matrix(embeddings)
            npz = io.BytesIO()
            np.savez_compressed(npz, embeddings=X,
                                subject_id=meta["subject_id"].to_numpy(),
                                session=meta["session"].to_numpy(),
                                site=meta["site"].to_numpy())
            z.writestr("embeddings.npz", npz.getvalue())
        z.writestr("README.txt",
                   "Synthetic multimodal biomedical dataset.\n"
                   "Generated by the CSP assessment data generator.\n"
                   "ground_truth.csv is oracle info for validating a prototype "
                   "on synthetic data; it would not exist on real data.\n")
    return buf.getvalue()


st.subheader("Download")
d1, d2, d3, d4, d5 = st.columns(5)
d1.download_button("clinical.csv", _csv(clinical), "clinical.csv", "text/csv",
                   use_container_width=True)
d2.download_button("embeddings.csv", _csv(embeddings), "embeddings.csv",
                   "text/csv", use_container_width=True)
d3.download_button("sensors.csv", _csv(sensors), "sensors.csv", "text/csv",
                   use_container_width=True)
d4.download_button("ground_truth.csv", _csv(ground_truth), "ground_truth.csv",
                   "text/csv", use_container_width=True)
d5.download_button("config.json",
                   json.dumps(data["config"], indent=2).encode("utf-8"),
                   "config.json", "application/json", use_container_width=True)

st.download_button("Download everything (ZIP bundle)", _bundle_zip(),
                   "synthetic_biomedical_dataset.zip", "application/zip",
                   type="primary")
