from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, pipeline
import nltk
import requests
import os
from dotenv import load_dotenv

nltk.download('stopwords')

# Initialize FastAPI
app = FastAPI()

load_dotenv()
GROQ_CLOUD_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")
if GROQ_CLOUD_API_KEY is None:
    raise ValueError("API key for Groq Cloud not found. Please set GROQ_CLOUD_API_KEY in your environment.")

# Define the input data model
class Review(BaseModel):
    review: str

def preprocess_review(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower().strip()
    return text

# METHOD 1: Load Logistic Regression model and vectorizer
model_lr = joblib.load('sentiment_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Preprocess function for Logistic Regression
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower().strip()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# METHOD 2: Load the Transformer model and tokenizer
model_name = './sentiment_transformer_model'  # Path where fine-tuned model is saved
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
transformer_model = DistilBertForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=transformer_model, tokenizer=tokenizer)

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use the /predict endpoint to get sentiment predictions."}


# Endpoint for method1: Logistic Regression
@app.post("/predict/method1")
async def predict_sentiment_method1(review: Review):
    # Preprocess and transform the input review
    processed_text = preprocess_text(review.review)
    transformed_text = tfidf.transform([processed_text])

    # Predict sentiment
    prediction = model_lr.predict(transformed_text)
    sentiment = "positive" if prediction[0] == "positive" else "negative"

    # Return the result
    return {"review": review.review, "sentiment": sentiment, "method": "Logistic Regression"}

# Endpoint for method2: Transformers (HuggingFace)
@app.post("/predict/method2")
async def predict_sentiment_method2(review: Review):
    # Use Hugging Face pipeline for prediction
    result = classifier(review.review)[0]
    sentiment = "positive" if result['label'] == 'LABEL_1' else "negative"

    # Return the result
    return {"review": review.review, "sentiment": sentiment, "method": "Transformers (HuggingFace)"}

def preprocess_review(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower().strip()
    return text

# METHOD 3: Using Groq Cloud's LLaMA 3 for Sentiment Analysis
@app.post("/predict/method3")
async def predict_sentiment_method3(review: Review):
    headers = {
        "Authorization": f"Bearer {GROQ_CLOUD_API_KEY}",
        "Content-Type": "application/json"
    }

    # Define the model name you intend to use
    model_name = "mixtral-8x7b-32768"  # Ensure this is the correct model ID from Groq documentation

    # Construct the messages for the chat API
    messages = [
        {"role": "user", "content": f"What is the sentiment of the following review: {review.review}"}
    ]

    # Construct the request payload
    data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 50,  # Limit response length
        "temperature": 0.7  # Adjust for randomness
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )

        print("Status code:", response.status_code)
        print("Response content as text:", response.text)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error from Groq Cloud API")

        result = response.json()
        sentiment = result["choices"][0]["message"]["content"].strip()  # Extract the sentiment

        return {
            "review": review.review,
            "sentiment": sentiment,
            "method": "LLaMA 3 via Groq Cloud"
        }

    except requests.exceptions.HTTPError as http_err:
        print("HTTP Error:", response.status_code, response.text)
        raise HTTPException(status_code=response.status_code, detail="Error from Groq Cloud API")
    except Exception as err:
        print("Other Error:", err)
        raise HTTPException(status_code=500, detail="Internal Server Error")