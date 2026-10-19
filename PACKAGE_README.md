# LoanLens Project

## Overview
LoanLens is an educational Streamlit application that predicts
historical US mortgage lending decisions using public HMDA data.

Selected model: Gradient Boosting (HistGB)
Calibration: Isotonic
Decision threshold: 0.60

These are historical-pattern predictions, not lending recommendations.

## Requirements
- Python 3.11 or a compatible version supported by requirements.txt
- Internet connection for downloading the dataset

## Setup (Windows PowerShell)
1. Extract this ZIP file.
2. Open a terminal in the extracted LoanLens folder.
3. Create a virtual environment:

   python -m venv .venv

4. Install dependencies:

   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

5. Set the source path:

   $env:PYTHONPATH = "src"

## Run the application

   .\.venv\Scripts\python.exe -m streamlit run app/app.py

Open the local URL printed in the terminal.

## Retrain the model
Run these commands in order:

   $env:PYTHONPATH = "src"
   .\.venv\Scripts\python.exe -m loanlens.download
   .\.venv\Scripts\python.exe -m loanlens.build_dataset
   .\.venv\Scripts\python.exe -m loanlens.train
   .\.venv\Scripts\python.exe -m loanlens.evaluate

## Run tests

   .\.venv\Scripts\python.exe -m pytest -q tests

## Important
Keep the artifacts folder intact. The application may depend on
the saved model, calibration, configuration, and evaluation files.

The app uses historical US mortgage data and is not validated for
Indian lending. Outputs require human review.
