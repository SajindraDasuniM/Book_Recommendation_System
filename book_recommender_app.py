
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') # added this line
import joblib
from sklearn.preprocessing import LabelEncoder


st.title("📚 Book Recommender")

# Load your model
model = joblib.load("best_model_svm.pkl")

# Load your vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#load trained meta data
train_data = joblib.load("train_data.pkl")

# load label encoder
le = joblib.load(le.pkl)

# Input box for book description
book_description = st.text_input("Enter a book description:")

def normalize_text(text):
    # Convert the text to lowercase
    text = text.lower()
    # Replace all non-word characters (anything other than a-z, A-Z, 0-9, and underscores) with a space
    text = re.sub(r'\W', ' ', text)
    # Replace one or more whitespace characters with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

text = normalize_text(book_description)

# preprocess the book description
import nltk

# Tokenization and stopwords removal
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize the input text into individual words
    tokens = word_tokenize(text)

    # Lemmatize each token and remove stop words
    # This creates a list of words that are lemmatized and are not in the stop words list
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join the tokens back into a single string separated by spaces
    return ' '.join(tokens)

# show recommendations
# Button to recommend
if st.button("Recommend"):
  if not book_description.strip():
        st.warning("Please enter a book description to get recommendations!")
  else:
    processed_text = preprocess_text(text)
    user_vector = vectorizer.transform([processed_text])

    st.write("Recommendations will appear here.")

    # Predict category using SVM
    subject = model.predict(user_vector)
    st.write(f"Predicted Subject: **{le.inverse_transform(subject)}**")

    # filter books in the same subject
    recommendations = train_data[train_data['Subject'] == le.inverse_transform(subject)[0]]
    st.write(f"Recommended books: **{recommendations}**")
