# CSP Prototype - three connected Streamlit apps

A pipeline from synthetic data -> a fitted demo model -> a presentation layer,
matching the assessment scenario (evolving model, multimodal biomedical data, no
agreed metric, cross-site heterogeneity, meaningful missingness).

## Files

| File | Role |
|---|---|
| `synthetic_data.py` | module: generates the synthetic multimodal dataset |
| `model_fit.py` | module: builds features, fits the demo model, produces the unified results contract (bootstrap variance + CI) |
| `app_common.py` | shared data-loader (upload prior step's ZIP/CSVs, or generate demo data) |
| `app_dataset_generator.py` | App 1: generate + download the dataset |
| `app_model_fit_generator.py` | App 2: fit demo models, export results.json / CSVs |
| `app_result_presenter.py` | App 3: 3-part presentation (variables -> data quality -> results) |
| `requirements.txt` | dependencies for all three |

The three apps are independent deployments but logically chained: App 1's ZIP is
the input to Apps 2 and 3. Apps 2 and 3 share model_fit.py, so the presenter
runs the fit live from its variable picker.

## Design stance (for the panel)

- The tool does not pick a model. It defines a model-agnostic results contract;
  logistic regression on embeddings -> diagnosis is only the demo estimator that
  fills it. App 2 also shows ridge regression and KMeans to prove the contract
  is not classifier-specific.
- Variance / CI come from the bootstrap, not a model-specific formula, so
  uncertainty is reported for any estimator and is honest about small n.
- Validation without an agreed metric = several lenses, each with a CI
  (accuracy, per-class sensitivity, confusion matrix, explicit failure list),
  rather than one headline number.

## Deploy each app on Streamlit Community Cloud

Put all files in one public GitHub repo. Create three apps, each pointing at a
different entry file: app_dataset_generator.py, app_model_fit_generator.py,
app_result_presenter.py.

## Run locally

    pip install -r requirements.txt
    streamlit run app_result_presenter.py

## Flow

1. App 1 - set subjects/sites/missingness, generate, download the ZIP.
2. App 2 - upload the ZIP (or generate demo data), pick variables, fit, download
   results.json + parameters.csv + predictions.csv.
3. App 3 - Part 1 pick variables; Part 2 missingness heatmap + descriptive
   stats; Part 3 parameter table (point / variance / CI), confusion matrix,
   per-class sensitivity, failure-case list.
