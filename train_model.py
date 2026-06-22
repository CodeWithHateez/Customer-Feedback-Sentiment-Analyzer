from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
DATA = BASE / "data" / "sample_reviews.csv"
MODEL = BASE / "models" / "sentiment_model.joblib"

def main():
    df = pd.read_csv(DATA)
    X = df["review"].astype(str)
    y = df["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL)
    print(f"Saved model to {MODEL}")

if __name__ == "__main__":
    main()
