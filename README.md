# LoanLens — Explainable Loan Application Approval Prediction

UCS321: AI for Engineers. This project is a supervised binary classifier trained on **real public US mortgage applications** from the CFPB/FFIEC HMDA Loan Application Register. It predicts whether lenders historically **Approved** or **Rejected** a comparable application, and it explains each prediction.

HDFC Bank is only the assignment's business framing. The project has no affiliation with HDFC, no HDFC data, and no validation for Indian lending. Amounts are in USD.

## Setup
The package includes a trained model, processed data, and evaluation reports, so
you can run the dashboard without downloading data or retraining. On Windows:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app/app.py
```

Stop any running dashboard before updating dependencies on Windows. Install the
full requirements file in the same Python environment used to start Streamlit;
`shap` is required for local explanations. The steps below are for rebuilding
the model from the original data.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src                                   # Windows PowerShell: $env:PYTHONPATH="src"
```

## 1. Get the data (official HMDA Data Browser API)
Automatic (needs internet access to `ffiec.cfpb.gov`):
```bash
python -m loanlens.download
```
Manual (if the automatic route is blocked): open each URL below in a browser, save the CSV, then register it.
```
https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=DE&years=2023&actions_taken=1,2,3
https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=DE&years=2024&actions_taken=1,2,3
```
```bash
python -m loanlens.download --register 2023 path/to/2023.csv --retrieved YYYY-MM-DD
python -m loanlens.download --register 2024 path/to/2024.csv --retrieved YYYY-MM-DD
```
Registered raw files are copied into `data/raw/` and made read-only. Their SHA-256 checksums go into `data/raw/provenance.json`.

## 2. Build → train → evaluate → app
```bash
python -m loanlens.build_dataset     # filters, target, features, audits, split manifest
python -m loanlens.train             # CV on 2023 train split; calibration + threshold on 2023 validation split
python -m loanlens.evaluate          # one pass on 2024 test; refuses silent reruns
python -m pytest -q tests
streamlit run app/app.py
```

## Layout
```
config.toml                 source, split, threshold-cost settings
src/loanlens/
  features.py               target rule, feature whitelist, leakage/protected/proxy lists, unit handling
  download.py               API download or manual registration, checksums, provenance
  build_dataset.py          schema validation, filters, missingness + duplicate audits, split
  modeling.py               ColumnTransformer + Pipeline, candidate models and grids
  train.py                  model selection, calibration, threshold
  evaluate.py               test metrics, figures, importance, error examples, subgroup audit
  predict.py / explain.py   saved-pipeline inference, schema checks, SHAP
app/app.py                  Streamlit dashboard
tests/                      leakage, units, schema, unseen categories, artifact loading, single/batch consistency
docs/                       dataset card, model card, limitations, report, slides, demo script, viva Q&A
```
