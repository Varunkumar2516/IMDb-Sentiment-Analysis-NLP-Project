
#  IMDb Movie Review Sentiment Analysis  

A **Machine Learning + Natural Language Processing (NLP)** project that classifies IMDb movie reviews as **Positive or Negative**.


It goes beyond a basic ML model by integrating a Human-in-the-Loop (HITL) feedback system, allowing the model to improve over time using real user corrections.
The project builds a complete NLP pipeline, compares multiple machine learning models, and deploys the best model using **Streamlit**.

---

##  Dataset Credit 
Credit = https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


## for live demo
#### [streamlit version](https://imdb-sentiment-analysis-nlp-project-ctzgpdtjx8tfhhkrbusfcb.streamlit.app/)

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
↓
Model IMprovement 
↓
Feedback from user
Retrain After Certain User Reviews
↓
90% accuracy of Logistic Regression
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

Text data was converted into numerical form using **TF-IDF Vectorization => 

##### 50000*10000
Baseline model 
max_features = 10000

##### 50000*45000**.
With this configuration
max_features = 45000
min_df = 10
max_df = 0.8
ngram_range = (1,2)

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
**Logistic Regression** achieved the best performance.

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
│── app.py # Streamlit application
│── retrain.py # Script to retrain model with feedback 
│── trained_models/ 
    │ ├── lr_model.pkl 
    │ └── tfidf_vectorizer.pkl 
│── feedback_data.csv # Stores user feedback 
│── Cleaned_Reviews_data.csv # Original dataset 
│── imbd_Dataset_data.csv # actual dataset 
│── requirements.txt 
│── README.md
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