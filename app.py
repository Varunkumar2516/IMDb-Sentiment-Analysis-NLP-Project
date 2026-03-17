import streamlit as st
import pickle
import re
import string
import contractions
import nltk
import pandas as pd
import os
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
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')

model = pickle.load(open("trained_models/lr_model.pkl","rb"))
vectorizer = pickle.load(open("trained_models/tfidf_vectorizer.pkl","rb"))

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

    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s+]', '', text)
    text = re.sub(r'\d+', '', text)

    # remove repeated chars
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    words = text.split()

    words = [w for w in words if w not in ActualStopwords]

    words = [w for w in words if (len(w) > 2 and len(w) < 15) or w in ['no']]

    lemmatized_words = []
    for word, tag in pos_tag(words):
        tag = get_wordnet_tag(tag)
        lemmatized_words.append(lemmatizer.lemmatize(word, tag))

    return " ".join(lemmatized_words)


#review,sentiment,Clean_reviews
def save_feedback(review, cleaned_text, label):

    df = pd.DataFrame(
        [[review, label, cleaned_text]],
        columns=['review', 'sentiment', 'Clean_reviews']
    )

    df.to_csv(
        "feedback_data.csv",
        mode='a',
        header=not os.path.exists("feedback_data.csv"),
        index=False
    )
st.title("🎬 Movie Review Sentiment Analysis")

st.write("Enter a movie review and the model will predict sentiment.")

review = st.text_area("Enter your review")


if "predicted" not in st.session_state:
    st.session_state.predicted = False

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned = NLP_pipeline(review)

        review_vec = vectorizer.transform([cleaned])
        prediction = model.predict(review_vec)[0]

        # store values
        st.session_state.predicted = True
        st.session_state.cleaned = cleaned
        st.session_state.prediction = prediction
        st.session_state.review = review


if st.session_state.predicted:

    st.write("Processed Text:", st.session_state.cleaned)

    if st.session_state.prediction == 'positive':
        st.success("😊 Positive Review")
    else:
        st.error("😞 Negative Review")

    st.subheader("Was the prediction correct?")

    feedback = st.selectbox(
        "Select option:",
        ["Select", "Correct", "Wrong"],
        key="feedback_select"
    )

    if feedback == "Correct":
        st.success("Thanks! 👍")

    elif feedback == "Wrong":

        correct_label = st.selectbox(
            "Select correct sentiment:",
            ["positive", "negative"],
            key="label_select"
        )

        if st.button("Submit Correction"):

            save_feedback(
                st.session_state.review,
                st.session_state.cleaned,
                correct_label
            )

            st.success("Feedback saved! Model will improve 🚀")

st.sidebar.title("TF-IDF Insights")

feature_names = vectorizer.get_feature_names_out()

if hasattr(model, "coef_"):

    coef = model.coef_[0]

    top_positive_idx = coef.argsort()[-20:]
    top_negative_idx = coef.argsort()[:20]

    pos_words = [(feature_names[i], coef[i]) for i in top_positive_idx]
    neg_words = [(feature_names[i], coef[i]) for i in top_negative_idx]

    st.sidebar.subheader("Top Positive Words")
    st.sidebar.dataframe(pd.DataFrame(pos_words, columns=["Word","Weight"]).sort_values("Weight", ascending=False))

    st.sidebar.subheader("Top Negative Words")
    st.sidebar.dataframe(pd.DataFrame(neg_words, columns=["Word","Weight"]).sort_values("Weight"))

# if os.path.exists("feedback_data.csv"):
#     df = pd.read_csv("feedback_data.csv")
#     st.sidebar.write(f"Feedback samples: {len(df)}")


# #we can do this manuualy Because providing this acces to Users can interupt the model and can be risky 
# if st.sidebar.button("Retrain Model"):
#     os.system("python retrain.py")
#     st.success("Model retrained!")