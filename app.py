from pathlib import Path
import joblib
import streamlit as st

BASE = Path(__file__).parent
MODEL = BASE / "models" / "sentiment_model.joblib"

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("Customer Feedback Sentiment Analyzer")
st.write("Paste a review and get a quick sentiment prediction.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

text = st.text_area("Enter feedback", height=160, placeholder="Type customer feedback here...")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        model = load_model()
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0]
        positive = round(float(prob[1]) * 100, 2)
        negative = round(float(prob[0]) * 100, 2)

        st.subheader("Result")
        st.success("Positive" if pred == 1 else "Negative")
        st.metric("Positive %", f"{positive}%")
        st.metric("Negative %", f"{negative}%")

        st.progress(int(positive))
