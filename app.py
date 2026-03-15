import streamlit as st
import pickle
import re
import string
import contractions
import nltk
import pandas as pd
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet


st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')


model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))

lemmatizer = WordNetLemmatizer()
ActualStopwords = set(stopwords.words('english')) - {'no','not','never','nor'}

def get_wordnet_tag(tag):

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def NLP_pipeline(text):

    html_pattern = re.compile(r'<.*?>')
    text = re.sub(html_pattern,' ',text)

    text = text.lower()

    text = contractions.fix(text)

    text = text.translate(str.maketrans('','',string.punctuation))

    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()

    words = [w for w in words if w not in ActualStopwords]

    lemmatized_words = []

    position_tag = pos_tag(words)

    for word, tag in position_tag:

        tag = get_wordnet_tag(tag)
        new_word = lemmatizer.lemmatize(word, tag)
        lemmatized_words.append(new_word)

    return " ".join(lemmatized_words)

st.title("🎬 Movie Review Sentiment Analysis")

st.write("Enter a movie review and the model will predict whether the sentiment is **positive or negative**.")


review = st.text_area(" Enter your review")

if st.button("Predict Sentiment"):

    cleaned = NLP_pipeline(review)

    review_vec = vectorizer.transform([cleaned])

    prediction = model.predict(review_vec)



    if prediction[0] == 'positive':

        st.success(f"😊 Positive Review ")

    else:

        st.error(f"😞 Negative Review )")

    st.write("Processed Text:", cleaned)


st.sidebar.title("TF-IDF Feature Insights")

st.sidebar.write("""
**TF-IDF (Term Frequency - Inverse Document Frequency)**  
measures how important a word is in a document relative to the dataset.

Higher TF-IDF score → word is more important.
""")

feature_names = vectorizer.get_feature_names_out()

# ---------------------------
# Model Feature Importance
# ---------------------------

if hasattr(model, "coef_"):

    coef = model.coef_[0]

    top_positive_idx = coef.argsort()[-20:]
    top_negative_idx = coef.argsort()[:20]

    positive_words = [(feature_names[i], coef[i]) for i in top_positive_idx]
    negative_words = [(feature_names[i], coef[i]) for i in top_negative_idx]

    pos_df = pd.DataFrame(positive_words, columns=["Word","Weight"])
    neg_df = pd.DataFrame(negative_words, columns=["Word","Weight"])

    st.sidebar.subheader("Top 20 Positive Words")

    st.sidebar.dataframe(pos_df.sort_values("Weight", ascending=False))

    st.sidebar.subheader("Top 20 Negative Words")

    st.sidebar.dataframe(neg_df.sort_values("Weight"))