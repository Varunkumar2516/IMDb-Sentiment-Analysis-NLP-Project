
#  IMDb Movie Review Sentiment Analysis

A **Machine Learning + Natural Language Processing (NLP)** project that classifies IMDb movie reviews as **Positive or Negative**.

The project builds a complete NLP pipeline, compares multiple machine learning models, and deploys the best model using **Streamlit**.

---

##  Dataset Credit 
Credit = https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


##  Project Workflow

```

Raw Text
↓
NLP Preprocessing
↓
TF-IDF Feature Extraction
↓
Model Training
↓
Model Evaluation
↓
Best Model Selection
↓
Streamlit Deployment

```

---

##  NLP Preprocessing Pipeline

The text data was cleaned and normalized using the following steps:

- HTML tag removal  
- Contraction expansion (don't → do not)  
- Lowercase conversion  
- Punctuation & special character removal  
- Tokenization  
- Stopword removal *(while keeping negations like not, no, never)*  
- Lemmatization  

Example:

```

Original:
"I wasn't impressed with this movie!!!"

Processed:
not impress movie

```

---

##  Feature Engineering

Text data was converted into numerical form using **TF-IDF Vectorization**.

TF-IDF measures how important a word is in a document relative to the entire dataset and is widely used for traditional NLP models.

---

##  Machine Learning Models Tested

| Model | Accuracy |
|------|------|
| Naive Bayes | ~0.85 |
| KNN | ~0.76 |
| Decision Tree | ~0.70 |
| Random Forest | ~0.85 |
| Logistic Regression | **0.891** |
| SVM | **0.893** |

###  Best Model
**Support Vector Machine (SVM)** achieved the best performance.

---

##  Model Evaluation

Models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

##  Word Importance Analysis

Using model coefficients, the project identifies words that strongly influence sentiment predictions.

**Positive words**

```

excellent
amazing
great
love
perfect

```

**Negative words**

```

terrible
worst
boring
bad
waste

```

---

##  Deployment

The trained model and TF-IDF vectorizer were saved using **pickle** and deployed with a **Streamlit web application**.

Run the app:

```

streamlit run app.py

```

Users can enter a movie review and get **real-time sentiment predictions**.

---

##  Project Structure

```

Sentiment-Analysis-IMDb
│
├── app.py
├── imdb_Dataset.csv
├── Cleaned_Reviews_data.csv
├── IMDB Sentiment Analysis.ipynb
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md

```

---

##  Technologies Used

- Python  
- Scikit-learn  
- NLTK  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- Streamlit  

---

##  Author

💖 *Created with heart by [Varun Kumar](https://github.com/Varunkumar2516)*   
B.Tech Computer Science Engineering  
DAVIET JALANDHAR

Interested in:

- Machine Learning  
- Natural Language Processing  
- AI Systems  
```
Helpfull for College students to Understand the NLP pipeline and Model Evaluation