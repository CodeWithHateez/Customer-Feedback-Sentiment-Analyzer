# Customer Feedback Sentiment Analyzer

A lightweight sentiment analysis project built for portfolio use and internship applications.

## What it does
- Trains a simple sentiment classifier on review text
- Predicts positive or negative sentiment
- Displays positive/negative percentages in a Streamlit UI
- Uses a small sample dataset so it stays easy to run on low-spec machines

## Tech stack
- Python
- pandas
- scikit-learn
- Streamlit
- joblib

## How to run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Resume bullet
Built a customer feedback sentiment analysis app in Python using scikit-learn and Streamlit, with model training, prediction output, and percentage-based review visualization.
