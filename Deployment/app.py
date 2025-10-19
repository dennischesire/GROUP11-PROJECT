# Deployment/app.py
# ======================================================================================
# Apple Product Sentiment Analyzer (Streamlit)
# - Robust artifact loading via pathlib (works regardless of CWD)
# - No NLTK 'punkt' / 'punkt_tab' dependency (uses wordpunct_tokenize)
# - Safe handling for models without predict_proba
# - Single + Batch analysis in tabs
# ======================================================================================

from __future__ import annotations

# ✅ Must be first Streamlit command
import streamlit as st
st.set_page_config(page_title="Apple Sentiment Analyzer", page_icon="🍎", layout="wide")

# ✅ Standard imports
from pathlib import Path
import re
import os
import joblib
import numpy as np
import pandas as pd

# ✅ NLP bits (no punkt)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize  # does NOT require 'punkt' / 'punkt_tab'

# ======================================================================================
# Paths / Artifacts
# ======================================================================================
APP_DIR = Path(__file__).resolve().parent
# If your PKLs are in Deployment/:
MODEL_PATH = APP_DIR / "sentiment_model.pkl"
VECT_PATH  = APP_DIR / "tfidf_vectorizer.pkl"

# If you keep them in Deployment/models/, uncomment:
# MODEL_PATH = APP_DIR / "models" / "sentiment_model.pkl"
# VECT_PATH  = APP_DIR / "models" / "tfidf_vectorizer.pkl"

# ======================================================================================
# NLTK setup (no 'punkt' / 'punkt_tab' downloads)
# ======================================================================================
@st.cache_data(show_spinner=False)
def setup_nltk() -> None:
    nltk_data_path = APP_DIR / "nltk_data"
    nltk.data.path.append(str(nltk_data_path))
    # Only what's needed for stopwords + lemmatizer
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, download_dir=str(nltk_data_path), quiet=True)
        except Exception:
            # Allow offline environments if data already present
            pass

setup_nltk()

# Preload stopwords/lemmatizer with graceful fallbacks
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    STOP_WORDS = set()
LEMMATIZER = WordNetLemmatizer()

# ======================================================================================
# Load model/vectorizer
# ======================================================================================
@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not VECT_PATH.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {VECT_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)

    # Ensure classes_ exists for labeling probabilities
    class_labels = getattr(model, "classes_", None)
    if class_labels is None:
        # Not fatal, but useful to surface early
        raise AttributeError("Loaded model has no attribute 'classes_'.")

    return model, vectorizer, list(class_labels)

# ======================================================================================
# Text preprocessing (no Punkt dependency)
# ======================================================================================
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # remove urls, mentions, hashtags
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)

    # tokenize without Punkt; splits words/punct safely
    tokens = wordpunct_tokenize(text)

    # keep only alphabetic tokens, remove very short noise
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1]

    # stopwords + lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]

    return " ".join(tokens)

# ======================================================================================
# Inference helpers
# ======================================================================================
def predict_single(text: str, model, vectorizer, class_labels: list[str]):
    processed = preprocess_text(text)
    if not processed:
        return {
            "processed": processed,
            "prediction": None,
            "probs": None,
            "warning": "Your input had no meaningful tokens after preprocessing."
        }

    X = vectorizer.transform([processed])
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        # Create a one-hot pseudo probability vector
        probs = np.zeros(len(class_labels), dtype=float)
        idx = class_labels.index(pred)
        probs[idx] = 1.0

    return {"processed": processed, "prediction": pred, "probs": probs, "warning": None}

def predict_batch(texts: pd.Series, model, vectorizer) -> np.ndarray:
    processed = texts.astype(str).apply(preprocess_text)
    X = vectorizer.transform(processed)
    preds = model.predict(X)
    return processed, preds

# ======================================================================================
# UI
# ======================================================================================
def main():
    st.title("🍎 Apple Product Sentiment Analyzer")
    st.markdown("Analyze tweets about Apple products and classify them as **Positive**, **Negative**, or **Neutral**.")

    # Load artifacts
    try:
        model, vectorizer, class_labels = load_model()
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        with st.expander("Debug paths"):
            st.code(f"MODEL_PATH = {MODEL_PATH}\nVECT_PATH = {VECT_PATH}")
        st.stop()

    st.sidebar.header("About the Model")
    st.sidebar.write(f"Classes: {class_labels}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**How it works:**")
    st.sidebar.markdown(
        "- Trained on 9,093 human-labeled tweets  \n"
        "- Uses ML to classify sentiment  \n"
        "- Focuses on Apple product discussions  \n"
        "- ~50% recall for negative feedback  "
    )

    tab_single, tab_batch, tab_info = st.tabs(["Single Prediction", "Batch Analysis", "Info"])

    # ------------------------- Single Prediction Tab -------------------------
    with tab_single:
        st.subheader("Analyze Tweet Sentiment")
        user_input = st.text_area(
            "Enter a tweet about Apple products:",
            placeholder="e.g., 'I love my new iPhone! The battery life is amazing.'",
            height=100
        )
        if st.button("Analyze Sentiment", type="primary"):
            if user_input.strip():
                result = predict_single(user_input, model, vectorizer, class_labels)
                if result["warning"]:
                    st.warning(result["warning"])

                pred = result["prediction"]
                if pred is None:
                    st.stop()

                sentiment_emojis = {
                    'Positive emotion': '😊 POSITIVE',
                    'Negative emotion': '😠 NEGATIVE',
                    'No emotion toward brand or product': '😐 NEUTRAL',
                    "I can't tell": '🤔 UNCLEAR'
                }
                sentiment_display = sentiment_emojis.get(str(pred), str(pred))

                st.subheader("Result")
                if 'POSITIVE' in sentiment_display:
                    st.success(f"**Sentiment:** {sentiment_display}")
                elif 'NEGATIVE' in sentiment_display:
                    st.error(f"**Sentiment:** {sentiment_display}")
                else:
                    st.info(f"**Sentiment:** {sentiment_display}")

                # Confidence section
                probs = result["probs"]
                if probs is not None:
                    conf_df = pd.DataFrame({
                        'Sentiment': class_labels,
                        'Confidence': (np.array(probs) * 100).astype(float)
                    })
                    st.subheader("Confidence Scores")
                    st.bar_chart(conf_df.set_index('Sentiment'))
                    st.dataframe(conf_df.style.format({'Confidence': '{:.1f}%'}))
                else:
                    st.info("Confidence scores not available for this model.")
            else:
                st.warning("Please enter some text to analyze.")

    # ------------------------- Batch Analysis Tab -------------------------
    with tab_batch:
        st.subheader("Batch Analysis (CSV)")
        uploaded_file = st.file_uploader("Upload CSV with a 'tweet_text' column", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()

            if 'tweet_text' not in df.columns:
                st.error("CSV must contain a 'tweet_text' column.")
            else:
                with st.spinner("Processing..."):
                    processed, preds = predict_batch(df['tweet_text'], model, vectorizer)
                    out_df = df.copy()
                    out_df['processed_text'] = processed
                    out_df['predicted_sentiment'] = preds

                st.success("Done!")
                st.dataframe(out_df, use_container_width=True)

                csv = out_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv, "sentiment_results.csv")

    # ------------------------- Info Tab -------------------------
    with tab_info:
        st.subheader("Model Performance (Static Summary)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Negative Recall", "50%", "↑ 5% vs target")
        c2.metric("Overall Accuracy", "62%")
        c3.metric("Tweets Analyzed", "9,093")

        st.markdown("---")
        st.markdown("Built with Streamlit | Apple Sentiment Analysis Project")

# Streamlit entrypoint
def main_entry():
    main()

if __name__ == "__main__":
    main_entry()
