import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="Apple Sentiment Analyzer", page_icon="🍎", layout="wide")

# ✅ Robust NLTK setup
@st.cache_data
def setup_nltk():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_path)

setup_nltk()

# ✅ Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# ✅ Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# ✅ Main app
def main():
    # Title and description
    st.title("🍎 Apple Product Sentiment Analyzer")
    st.markdown("""
    This tool analyzes tweets about Apple products and classifies them as **Positive**, **Negative**, or **Neutral** sentiment.
    """)

    # Load model
    try:
        model, vectorizer = load_model()
        class_labels = model.classes_
        st.sidebar.write(f"Model classes: {class_labels}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Layout
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
                text_vectorized = vectorizer.transform([processed_text])
                prediction = model.predict(text_vectorized)[0]
                probability = model.predict_proba(text_vectorized)[0]

                st.subheader("Results")
                sentiment_emojis = {
                    'Positive emotion': '😊 POSITIVE',
                    'Negative emotion': '😠 NEGATIVE',
                    'No emotion toward brand or product': '😐 NEUTRAL',
                    "I can't tell": '🤔 UNCLEAR'
                }
                sentiment_display = sentiment_emojis.get(prediction, prediction)

                if 'POSITIVE' in sentiment_display:
                    st.success(f"**Sentiment:** {sentiment_display}")
                elif 'NEGATIVE' in sentiment_display:
                    st.error(f"**Sentiment:** {sentiment_display}")
                else:
                    st.info(f"**Sentiment:** {sentiment_display}")

                st.subheader("Confidence Scores")
                conf_df = pd.DataFrame({
                    'Sentiment': class_labels,
                    'Confidence': probability * 100
                })
                st.bar_chart(conf_df.set_index('Sentiment'))
                st.write("Detailed probabilities:")
                st.dataframe(conf_df.style.format({'Confidence': '{:.1f}%'}))
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

# ✅ Run app
if __name__ == "__main__":
    main()

# ✅ Batch analysis section
st.subheader("Batch Analysis")
uploaded_file = st.file_uploader("Upload CSV with tweets", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'tweet_text' in df.columns:
        model, vectorizer = load_model()
        processed_texts = df['tweet_text'].apply(preprocess_text)
        predictions = model.predict(vectorizer.transform(processed_texts))
        df['predicted_sentiment'] = predictions
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "sentiment_results.csv")
    else:
        st.error("CSV must contain a 'tweet_text' column.")