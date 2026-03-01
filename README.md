# Book_Recommendation_System
Built an end-to-end book recommender system using scikit-learn, Streamlit and NLP to predict the subject based on a given description and suggest similar textbooks for schools. Used GoogleBook and OpenLibrary APIs to collect metadata. Implemented text preprocessing: tokenization, stopword removal, and lemmatization using NLTK. Trained three models: NN(unsupervised), KNN and SVM to predict the subject. SVM showed the highest(83%) accuracy in predicting subject. Implemented a recommender to suggest relevant textbooks based on predicted subject. Created and pickled trained models (SVM, vectorizer, LabelEncoder) for deployment. Designed a user-friendly interactive web app using Streamlit, displaying predicted book subjects and recommended titles. Deployed the app online via Streamlit Cloud, integrated with GitHub for version control.
Link to the app:

https://bookrecommendationsystem-exktjt5tgkkrzglzaawgbe.streamlit.app/
