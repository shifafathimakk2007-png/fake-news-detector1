import streamlit as st
import joblib

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
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)
        st.write("Prediction:", prediction[0])

        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")
    else:
        st.warning("Please enter some news.")
