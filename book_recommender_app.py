
import streamlit as st
import pickle
import pandas as pd
import nltk
nltk.download('punkt_tab') # added this line
from sklearn.preprocessing import LabelEncoder

st.title("📚 Book Recommender")

# Load your model
with open("best_model_svm.pkl", "rb") as f:
  model = pickle.load(f)

# Load your vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

#load trained meta data
with open("train_data.pkl", "rb") as f:
    train_data = pickle.load(f)

# Input box for book description
book_description = st.text_input("Enter a book description:")

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

processed_text = preprocess_text(book_description)

#transform the user input
user_vector = vectorizer.transform([processed_text])

# Example usage
n = 20
description_to_recommend = test_data['combined_text'].iloc[n] # select the description to recommend books
print("Selected description from test data")
print(test_data.iloc[n])
print("")

recommendations = get_recommendations_svm(description_to_recommend, best_model_svm, tfidf_vectorizer, train_data)
print("")
print(f"Recommendations: {recommendations[['ISBN', 'title', 'School_ID', 'Year', 'Subject']]}")

# Button to recommend
if st.button("Recommend"):
    st.write("Recommendations will appear here.")
    # Predict category using SVM
    subject = model.predict(user_vector)
    st.write(f"Predicted Subject: **{{le.inverse_transform(subject)}}**")
    recommendations = train_data[train_data['Subject'] == le.inverse_transform(subject)[0]]
