"""
synthetic_data.py
=================
Synthetic multimodal biomedical dataset generator for the CSP "Research Data &
AI Solutions Specialist" assessment.

Why this exists
---------------
The scenario describes an early-stage research setting: an evolving ML model on
high-dimensional biomedical data, no agreed evaluation metric, and data that
varies across sites in modality, scale, quality, and completeness. To prototype
a tool against that world, you need data that *looks like* that world. Real data
can't be shared and doesn't exist yet, so we synthesize it with the same
structural pain points baked in.

The design choices below each map to a requirement in the task:

1. MULTIMODAL  -> every subject-session may carry:
     - a high-dimensional model EMBEDDING (the "feature space / model output")
     - a wearable-style SENSOR time series (the "multimodal sensor output")
     - TABULAR clinical + behavioral data
2. LONGITUDINAL -> multiple sessions per subject (default 2) with a progression
   effect for affected classes, so longitudinal views are meaningful.
3. CROSS-SITE HETEROGENEITY -> sites differ in: a batch/scale shift applied to
   embeddings, sensor sampling rate (so time series have different lengths),
   measurement noise (quality), which modalities they even collect, which label
   schema they use (diagnosis vs. behavioral score), and dropout rate.
4. MEANINGFUL MISSINGNESS -> not MCAR. It is:
     - STRUCTURAL: a site that never collects sensors (whole modality absent),
     - MNAR-ish DROPOUT: higher-severity subjects more likely to skip session 2,
     - MAR: clinical fields missing at a site-dependent rate.
5. EVOLVING MODEL -> `model_version` reseeds the embedding projection and noise,
   so the SAME subject produces DIFFERENT embeddings across versions. This is
   what lets a downstream tool demonstrate "how outputs change under different
   parameters / model revisions."
6. RECOVERABLE GROUND TRUTH -> all modalities are driven by a shared low-dim
   latent factor + class. Because the generative truth is known, a prototype can
   be *validated on synthetic data* (does it recover the planted structure?)
   even though no agreed metric exists on the real data. That is the bridge to
   the "validation without an agreed-upon metric" requirement.

Nothing here is implementation-specific to a model; it produces portable tables
and arrays (CSV / NPZ) that any prototype can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    """All knobs for one generation run. Defaults mirror the scenario's
    concrete example (30 subjects, 2 sessions, meaningful missingness)."""

    n_subjects: int = 30
    n_sessions: int = 2
    n_sites: int = 3
    n_classes: int = 3          # e.g. Control / Condition-A / Condition-B
    embedding_dim: int = 128    # high-dimensional model output
    latent_dim: int = 6         # hidden factor driving every modality
    sensor_channels: int = 3
    sensor_seconds: float = 4.0
    progression: float = 0.6    # session-over-session drift for affected classes
    missingness: float = 1.0    # global multiplier on all missingness rates
    class_sep: float = 1.0      # between-class separation in latent space
    class_spread: float = 1.7   # within-class spread (overlap -> imperfect models)
    model_version: int = 1      # reseeds embedding projection + noise
    seed: int = 7

    # site label schemas: which targets each site records
    #   "both"        -> diagnosis + behavioral score
    #   "diagnosis"   -> categorical only
    #   "behavioral"  -> continuous score only
    site_label_schema: tuple = ("both", "diagnosis", "behavioral")

    # site sensor sampling rates in Hz; None means the site never records sensors
    site_sensor_fs: tuple = (64, 32, None)

    # per-site quality (measurement noise multiplier; higher = noisier/worse)
    site_noise: tuple = (0.8, 1.2, 1.8)

    # per-site session-2 dropout base rate and clinical-missing base rate
    site_dropout: tuple = (0.05, 0.15, 0.30)
    site_clinical_missing: tuple = (0.03, 0.10, 0.20)

    def __post_init__(self):
        # pad/truncate per-site tuples so config stays valid for any n_sites
        def fit(t, fill):
            t = list(t)
            if len(t) < self.n_sites:
                t = t + [t[-1] if t else fill] * (self.n_sites - len(t))
            return tuple(t[: self.n_sites])

        self.site_label_schema = fit(self.site_label_schema, "both")
        self.site_sensor_fs = fit(self.site_sensor_fs, 50)
        self.site_noise = fit(self.site_noise, 1.0)
        self.site_dropout = fit(self.site_dropout, 0.1)
        self.site_clinical_missing = fit(self.site_clinical_missing, 0.1)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def _class_means(rng: np.random.Generator, n_classes: int, latent_dim: int,
                 class_sep: float = 1.0) -> np.ndarray:
    """Class centroids in latent space. `class_sep` controls how far apart they
    sit; combined with within-class spread it sets how separable the classes are
    (and therefore how imperfect a simple model will be)."""
    means = rng.normal(0, 1, size=(n_classes, latent_dim))
    means *= class_sep
    return means


def _embedding_projection(seed: int, model_version: int, latent_dim: int,
                          embedding_dim: int):
    """Projection from latent -> embedding space. Reseeded by model_version so a
    new 'model revision' yields a different but related embedding geometry."""
    rng = np.random.default_rng((seed + 1) * 1000 + model_version)
    W = rng.normal(0, 1.0 / np.sqrt(latent_dim), size=(latent_dim, embedding_dim))
    # version-dependent global noise scale on embeddings
    emb_noise = 0.25 + 0.15 * ((model_version - 1) % 3)
    class_offset_scale = 0.8
    return W, emb_noise, class_offset_scale


def _make_sensor(rng, z, fs, seconds, channels, noise):
    """Wearable-style multichannel signal whose frequency + amplitude depend on
    the latent factor z. Returns (t, signal[channels, T])."""
    T = int(round(fs * seconds))
    t = np.arange(T) / fs
    sig = np.zeros((channels, T))
    for c in range(channels):
        base_f = 1.0 + 2.5 * abs(z[c % len(z)])          # latent-driven frequency
        amp = 0.6 + 0.5 * abs(z[(c + 1) % len(z)])       # latent-driven amplitude
        phase = rng.uniform(0, 2 * np.pi)
        sig[c] = amp * np.sin(2 * np.pi * base_f * t + phase)
        sig[c] += 0.3 * amp * np.sin(2 * np.pi * (2 * base_f) * t)  # harmonic
        sig[c] += rng.normal(0, 0.25 * noise, size=T)               # site noise
    return t, sig


def generate(config: Optional[GenConfig] = None):
    """Generate the full synthetic dataset.

    Returns a dict of artifacts:
      clinical    : DataFrame, one row per existing (subject, session)
      embeddings  : DataFrame, subject/session/site/model_version + emb_0..N
      sensors     : long DataFrame [subject_id, session, site, channel, t, value]
      ground_truth: DataFrame, subject-level oracle (true class + latent factor)
      config      : the GenConfig used (as dict)
    """
    cfg = config or GenConfig()
    rng = np.random.default_rng(cfg.seed)

    class_means = _class_means(rng, cfg.n_classes, cfg.latent_dim, cfg.class_sep)
    W, emb_noise, class_off_scale = _embedding_projection(
        cfg.seed, cfg.model_version, cfg.latent_dim, cfg.embedding_dim
    )
    # fixed class offsets in embedding space (shared across sessions)
    class_emb_offset = rng.normal(0, class_off_scale, size=(cfg.n_classes, cfg.embedding_dim))

    # per-site batch effects on embeddings (the "no unified format / shift" pain)
    site_batch = rng.normal(0, 0.6, size=(cfg.n_sites, cfg.embedding_dim))
    site_scale = rng.uniform(0.85, 1.25, size=cfg.n_sites)

    clinical_rows = []
    emb_rows = []
    sensor_rows = []
    truth_rows = []

    diag_labels = ["Control", "Condition-A", "Condition-B", "Condition-C",
                   "Condition-D"][: cfg.n_classes]

    for s in range(cfg.n_subjects):
        site = int(rng.integers(0, cfg.n_sites))
        schema = cfg.site_label_schema[site]
        fs = cfg.site_sensor_fs[site]
        noise = cfg.site_noise[site]

        true_class = int(rng.integers(0, cfg.n_classes))
        # latent factor for this subject around its class centroid
        z0 = class_means[true_class] + rng.normal(0, cfg.class_spread, size=cfg.latent_dim)
        # severity only meaningful for non-control classes; drives dropout + drift
        severity = 0.0 if true_class == 0 else float(np.clip(rng.normal(0.6, 0.25), 0, 1))

        age = int(np.clip(rng.normal(42, 14), 18, 85))
        sex = rng.choice(["F", "M"])
        subject_id = f"S{site}-{s:03d}"

        truth_rows.append({
            "subject_id": subject_id,
            "site": site,
            "true_class": true_class,
            "true_diagnosis": diag_labels[true_class],
            "true_severity": round(severity, 3),
            **{f"z_{i}": round(float(z0[i]), 4) for i in range(cfg.latent_dim)},
        })

        for sess in range(cfg.n_sessions):
            # ---- dropout: session 0 always present; later sessions can drop ---
            if sess > 0:
                p_drop = cfg.site_dropout[site] * (1 + 1.5 * severity) * cfg.missingness
                if rng.random() < min(p_drop, 0.9):
                    continue  # subject missed this session entirely (MNAR-ish)

            # progression: affected classes drift in latent space over sessions
            z = z0 + (sess * cfg.progression * severity) * (class_means[true_class] /
                                                            (np.linalg.norm(class_means[true_class]) + 1e-9))
            days = int(sess * (90 + rng.integers(-10, 10)))  # ~quarterly sessions

            # ---- embedding modality (may be structurally/clinically missing) ---
            has_emb = True
            # small chance a present session is missing its embedding (upload gap)
            if rng.random() < 0.05 * cfg.missingness:
                has_emb = False
            if has_emb:
                emb = z @ W
                emb = emb + class_emb_offset[true_class]
                emb = emb * site_scale[site] + site_batch[site]
                emb = emb + rng.normal(0, emb_noise * noise, size=cfg.embedding_dim)
                row = {"subject_id": subject_id, "session": sess, "site": site,
                       "model_version": cfg.model_version}
                row.update({f"emb_{i}": float(emb[i]) for i in range(cfg.embedding_dim)})
                emb_rows.append(row)

            # ---- sensor modality (whole-modality structural missingness) -------
            has_sensor = fs is not None
            if has_sensor and rng.random() < 0.07 * cfg.missingness:
                has_sensor = False  # occasional per-session sensor gap
            if has_sensor:
                t, sig = _make_sensor(rng, z, fs, cfg.sensor_seconds,
                                      cfg.sensor_channels, noise)
                for c in range(cfg.sensor_channels):
                    for ti in range(len(t)):
                        sensor_rows.append({
                            "subject_id": subject_id, "session": sess, "site": site,
                            "channel": c, "fs": fs,
                            "t": round(float(t[ti]), 4),
                            "value": round(float(sig[c, ti]), 5),
                        })

            # ---- tabular clinical + behavioral targets -------------------------
            # behavioral score is a noisy linear readout of the latent factor
            behavioral = float(np.dot(z, np.linspace(1, -1, cfg.latent_dim)) +
                               rng.normal(0, 0.8))
            diagnosis = diag_labels[true_class]

            # apply site label schema (heterogeneous label availability)
            if schema == "diagnosis":
                behavioral = np.nan
            elif schema == "behavioral":
                diagnosis = np.nan

            # MAR clinical missingness (site-dependent rate)
            cm = cfg.site_clinical_missing[site] * cfg.missingness
            age_v = age if rng.random() > cm else np.nan
            sex_v = sex if rng.random() > cm else np.nan
            if not (isinstance(behavioral, float) and np.isnan(behavioral)):
                if rng.random() < cm:
                    behavioral = np.nan

            quality = float(np.clip(rng.normal(1.0 / noise, 0.1), 0.2, 1.5))

            clinical_rows.append({
                "subject_id": subject_id,
                "session": sess,
                "site": site,
                "site_label_schema": schema,
                "days_since_baseline": days,
                "age": age_v,
                "sex": sex_v,
                "diagnosis": diagnosis,
                "behavioral_score": (np.nan if (isinstance(behavioral, float) and
                                                np.isnan(behavioral))
                                     else round(behavioral, 3)),
                "quality_score": round(quality, 3),
                "has_embedding": has_emb,
                "has_sensor": has_sensor,
            })

    clinical = pd.DataFrame(clinical_rows)
    embeddings = pd.DataFrame(emb_rows)
    sensors = pd.DataFrame(sensor_rows)
    ground_truth = pd.DataFrame(truth_rows)

    return {
        "clinical": clinical,
        "embeddings": embeddings,
        "sensors": sensors,
        "ground_truth": ground_truth,
        "config": asdict(cfg),
    }


# --------------------------------------------------------------------------- #
# Small helpers used by the app / downstream tools
# --------------------------------------------------------------------------- #
def embedding_matrix(embeddings: pd.DataFrame):
    """Return (meta_df, X) where X is the float embedding matrix."""
    emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
    meta = embeddings[["subject_id", "session", "site", "model_version"]].reset_index(drop=True)
    X = embeddings[emb_cols].to_numpy()
    return meta, X


def pca_2d(X: np.ndarray):
    """Lightweight PCA via SVD (no sklearn dependency) -> Nx2."""
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2))
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :2] * S[:2])


def missingness_summary(clinical: pd.DataFrame, cfg_n_subjects: int,
                        cfg_n_sessions: int) -> pd.DataFrame:
    """Per-site completeness view, the kind of thing a tool would surface."""
    g = clinical.groupby("site")
    rows = []
    for site, sub in g:
        rows.append({
            "site": site,
            "sessions_present": len(sub),
            "subjects": sub["subject_id"].nunique(),
            "embedding_coverage": round(sub["has_embedding"].mean(), 3),
            "sensor_coverage": round(sub["has_sensor"].mean(), 3),
            "diagnosis_present": round(sub["diagnosis"].notna().mean(), 3),
            "behavioral_present": round(sub["behavioral_score"].notna().mean(), 3),
            "mean_quality": round(sub["quality_score"].mean(), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # quick smoke test
    data = generate(GenConfig())
    print("clinical:", data["clinical"].shape)
    print("embeddings:", data["embeddings"].shape)
    print("sensors:", data["sensors"].shape)
    print("ground_truth:", data["ground_truth"].shape)
    print(missingness_summary(data["clinical"], 30, 2))
