# Customer Feedback Sentiment Analyzer

A lightweight sentiment analysis project built with Python and Streamlit that classifies customer review text as positive or negative and displays sentiment percentages in an easy-to-use web interface.

## What it does
- Trains a simple sentiment classifier on review text
- Predicts whether feedback is positive or negative
- Shows positive and negative sentiment percentages in a Streamlit dashboard
- Uses a small sample dataset so it runs well on low-spec machines

## Tech Stack
- Python
- pandas
- scikit-learn
- Streamlit
- joblib

## How to Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py

