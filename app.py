import streamlit as st
import joblib

import re
import string
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text
# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Website title
st.title("Fake News Detection System")
st.write("Check whether a news article is Fake or Real.")

# News input
news = st.text_area("Enter News Here:")

# Check button
if st.button("Check News"):
    if news.strip():
        cleaned_news = clean_text(news)
news_vector = vectorizer.transform([cleaned_news])
        prediction = model.predict(news_vector)
        st.write("Prediction:", prediction[0])

        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")
    else:
        st.warning("Please enter some news.")
