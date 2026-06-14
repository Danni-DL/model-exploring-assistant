"""
app_common.py
=============
Shared data loader used by the model-fit app and the presenter app. Because the
three apps are separate Streamlit Cloud deployments, the connective tissue is
files: the dataset generator exports a ZIP (or CSVs); downstream apps upload it.
Each downstream app can ALSO generate demo data on the fly, so it runs
standalone in an interview without any manual upload step.
"""

import io
import zipfile

import pandas as pd
import streamlit as st

import synthetic_data as sd


@st.cache_data(show_spinner="Generating demo data...")
def _demo(n_subjects: int, seed: int, missingness: float):
    cfg = sd.GenConfig(n_subjects=n_subjects, seed=seed, missingness=missingness)
    d = sd.generate(cfg)
    return d["clinical"], d["embeddings"], d["ground_truth"]


def _read_zip(file):
    """Extract clinical/embeddings/ground_truth CSVs from a generator ZIP."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(file.read())) as z:
        for name in z.namelist():
            base = name.split("/")[-1]
            if base in ("clinical.csv", "embeddings.csv", "ground_truth.csv"):
                out[base.replace(".csv", "")] = pd.read_csv(z.open(name))
    return out


def load_dataset_ui(key: str = ""):
    """Render data-source controls and return (clinical, embeddings,
    ground_truth) or (None, None, None)."""
    st.subheader("1. Data source")
    src = st.radio(
        "Where does the data come from?",
        ["Generate demo data", "Upload from dataset generator"],
        horizontal=True, key=f"src_{key}",
    )

    if src == "Generate demo data":
        c1, c2, c3 = st.columns(3)
        n = c1.slider("Subjects", 20, 200, 60, 10, key=f"n_{key}",
                      help="More subjects -> more rows to fit and tighter CIs.")
        seed = c2.number_input("Seed", 0, 9999, 1, 1, key=f"seed_{key}")
        miss = c3.slider("Missingness", 0.0, 2.0, 1.0, 0.1, key=f"miss_{key}")
        clinical, embeddings, ground_truth = _demo(n, seed, miss)
        st.caption(f"Generated {clinical.shape[0]} session rows across "
                   f"{ground_truth.shape[0]} subjects.")
        return clinical, embeddings, ground_truth

    st.caption("Upload the ZIP bundle from the dataset generator, or the three "
               "CSVs individually.")
    zf = st.file_uploader("ZIP bundle", type="zip", key=f"zip_{key}")
    if zf is not None:
        parts = _read_zip(zf)
        if all(k in parts for k in ("clinical", "embeddings", "ground_truth")):
            return parts["clinical"], parts["embeddings"], parts["ground_truth"]
        st.error("ZIP is missing one of clinical/embeddings/ground_truth.csv")
        return None, None, None

    c1, c2, c3 = st.columns(3)
    cf = c1.file_uploader("clinical.csv", type="csv", key=f"cf_{key}")
    ef = c2.file_uploader("embeddings.csv", type="csv", key=f"ef_{key}")
    gf = c3.file_uploader("ground_truth.csv", type="csv", key=f"gf_{key}")
    if cf and ef and gf:
        return pd.read_csv(cf), pd.read_csv(ef), pd.read_csv(gf)

    st.info("Waiting for data...")
    return None, None, None
