# This is your Streamlit app file
import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Clean text function
def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text

# Prediction function
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Positive 😊" if prediction == 1 else "Negative 😞"

# Streamlit UI
st.title("📝 Sentiment Analysis App")
review = st.text_input("Enter a review to find out its sentiment:")
if st.button("Predict"):
    result = predict_sentiment(review)
    st.success(f"The review is {result}")
