import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import json

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

# ---------------------------
# Load Tokenizer for LSTM
# ---------------------------
def load_tokenizer(path='models/saved_models/tokenizer.json'):
    with open(path, 'r') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

# ---------------------------
# Load LSTM Model
# ---------------------------
@st.cache_resource
def load_lstm_model():
    model = tf.keras.models.load_model("models/saved_models/lstm_model.h5")
    tokenizer = load_tokenizer()
    return model, tokenizer

# ---------------------------
# Load Logistic Regression Model
# ---------------------------
@st.cache_resource
def load_logistic_model():
    model = joblib.load("models/saved_models/logistic_model.pkl")
    vectorizer = joblib.load("models/saved_models/tfidf_vectorizer.pkl")
    return model, vectorizer

# ---------------------------
# Predict with LSTM
# ---------------------------
def predict_with_lstm(text, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)
    class_idx = np.argmax(prediction)
    sentiment = ['Negative', 'Neutral', 'Positive'][class_idx]
    return sentiment, prediction[0]

# ---------------------------
# Predict with Logistic Regression
# ---------------------------
def predict_with_logistic(text, model, vectorizer):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return prediction, None

# ---------------------------
# Streamlit Web App
# ---------------------------
def main():
    st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬")
    st.title("💬 Tweet Sentiment Analyzer")
    st.subheader("Classify tweet as Positive, Negative, or Neutral")

    model_type = st.selectbox("Choose Model:", ["LSTM", "Logistic Regression"])

    tweet = st.text_area("Enter a tweet:", placeholder="Type a tweet...")

    if st.button("Analyze Sentiment"):
        if tweet.strip() == "":
            st.warning("Please enter a tweet to analyze.")
        else:
            if model_type == "LSTM":
                model, tokenizer = load_lstm_model()
                sentiment, confidence = predict_with_lstm(tweet, model, tokenizer)
                st.success(f"Predicted Sentiment (LSTM): **{sentiment}**")
                st.write("Confidence Scores:")
                st.json({
                    "Negative": f"{confidence[0]*100:.2f}%",
                    "Neutral": f"{confidence[1]*100:.2f}%",
                    "Positive": f"{confidence[2]*100:.2f}%"
                })
            else:
                model, vectorizer = load_logistic_model()
                sentiment, _ = predict_with_logistic(tweet, model, vectorizer)
                st.success(f"Predicted Sentiment (Logistic Regression): **{sentiment}**")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit | By Tilak")

if __name__ == '__main__':
    main()
