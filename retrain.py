
import pandas as pd
import pickle

original = pd.read_csv("Cleaned_Reviews_data.csv")
feedback = pd.read_csv("feedback_data.csv")

data = pd.concat([original, feedback])

vectorizer = pickle.load(open("trained_models/tfidf_vectorizer.pkl","rb"))
model = pickle.load(open("trained_models/lr_model.pkl","rb"))

X = vectorizer.fit_transform(data['Clean_reviews'])
y = data['sentiment']
sample_weight = [1]*len(original) + [2]*len(feedback)

model.fit(X, y, sample_weight=sample_weight)


pickle.dump(model, open("trained_models/lr_model3.pkl","wb"))