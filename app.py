# ✅ Imports
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize  # <- does NOT require punkt/punkt_tab

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Apple Sentiment Analyzer", page_icon="🍎", layout="wide")

# ✅ NLTK setup (no punkt / no punkt_tab)
@st.cache_data
def setup_nltk():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    # Only the resources we actually need
    resources = ["stopwords", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            nltk.download(r, download_dir=nltk_data_path, quiet=True)
        except Exception:
            # Allow offline environments to proceed if data already present
            pass

setup_nltk()

# ✅ Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# ✅ Text preprocessing (no Punkt dependency)
def preprocess_text(text: str) -> str:
    text = text.lower()
    # remove urls, mentions, hashtags
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)

    # tokenize without Punkt; split on words/punct
    tokens = wordpunct_tokenize(text)

    # keep only alphabetic tokens and remove very short noise
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1]

    # stopwords + lemmatize
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()  # fallback if offline and missing data

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]

    return " ".join(tokens)

# ✅ Main app
def main():
    st.title("🍎 Apple Product Sentiment Analyzer")
    st.markdown("""
    Analyze tweets about Apple products and classify them as **Positive**, **Negative**, or **Neutral**.
    """)

    # Load model
    try:
        model, vectorizer = load_model()
        class_labels = getattr(model, "classes_", None)
        if class_labels is None:
            raise AttributeError("Model has no attribute 'classes_'.")
        st.sidebar.write(f"Model classes: {list(class_labels)}")
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Analyze Tweet Sentiment")
        user_input = st.text_area(
            "Enter a tweet about Apple products:",
            placeholder="e.g., 'I love my new iPhone! The battery life is amazing.'",
            height=100
        )

        if st.button("Analyze Sentiment", type="primary"):
            if user_input.strip():
                processed_text = preprocess_text(user_input)
                if not processed_text:
                    st.warning("Your input had no meaningful tokens after preprocessing.")
                # Vectorize & predict
                try:
                    text_vectorized = vectorizer.transform([processed_text])
                    prediction = model.predict(text_vectorized)[0]
                    # Some models may lack predict_proba; handle gracefully
                    if hasattr(model, "predict_proba"):
                        probability = model.predict_proba(text_vectorized)[0]
                    else:
                        # Create a pseudo-probability: 1.0 for predicted class, 0 for others
                        probability = np.zeros(len(class_labels), dtype=float)
                        probability[list(class_labels).index(prediction)] = 1.0
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.stop()

                st.subheader("Results")
                sentiment_emojis = {
                    'Positive emotion': '😊 POSITIVE',
                    'Negative emotion': '😠 NEGATIVE',
                    'No emotion toward brand or product': '😐 NEUTRAL',
                    "I can't tell": '🤔 UNCLEAR'
                }
                sentiment_display = sentiment_emojis.get(str(prediction), str(prediction))

                if 'POSITIVE' in sentiment_display:
                    st.success(f"**Sentiment:** {sentiment_display}")
                elif 'NEGATIVE' in sentiment_display:
                    st.error(f"**Sentiment:** {sentiment_display}")
                else:
                    st.info(f"**Sentiment:** {sentiment_display}")

                st.subheader("Confidence Scores")
                try:
                    conf_df = pd.DataFrame({
                        'Sentiment': list(class_labels),
                        'Confidence': (probability * 100).astype(float)
                    })
                    st.bar_chart(conf_df.set_index('Sentiment'))
                    st.write("Detailed probabilities:")
                    st.dataframe(conf_df.style.format({'Confidence': '{:.1f}%'}))
                except Exception:
                    st.info("Confidence scores not available for this model.")
            else:
                st.warning("Please enter some text to analyze.")

    with col2:
        st.subheader("About This Tool")
        st.markdown("""
        **How it works:**
        - Trained on 9,093 human-labeled tweets  
        - Uses machine learning to classify sentiment  
        - Focuses on Apple product discussions  
        - 50% accuracy in detecting negative feedback  
        """)
        st.subheader("Model Performance")
        st.metric("Negative Recall", "50%", "5% above target")
        st.metric("Overall Accuracy", "62%")
        st.metric("Tweets Analyzed", "9,093")

    st.markdown("---")
    st.markdown("Built with Streamlit | Apple Sentiment Analysis Project")

main()

# ✅ Batch analysis
st.subheader("Batch Analysis")
uploaded_file = st.file_uploader("Upload CSV with tweets", type=['csv'])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if 'tweet_text' in df.columns:
        try:
            model, vectorizer = load_model()
            processed_texts = df['tweet_text'].astype(str).apply(preprocess_text)
            preds = model.predict(vectorizer.transform(processed_texts))
            df['predicted_sentiment'] = preds

            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "sentiment_results.csv")
        except Exception as e:
            st.error(f"Batch inference error: {e}")
    else:
        st.error("CSV must contain a 'tweet_text' column.")
