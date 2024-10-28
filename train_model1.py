import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load and preprocess the dataset
df = pd.read_csv('dataset.csv')

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower().strip()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

df['review'] = df['review'].apply(preprocess_text)

# Transform the data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['review'])
y = df['sentiment']

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_tfidf, y)

# Save the trained model and TF-IDF vectorizer
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

print("Model and vectorizer saved!")
