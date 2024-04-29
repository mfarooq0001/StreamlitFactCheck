from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
model = joblib.load('../model/fake_news_model.pkl')
tfidf_vectorizer = joblib.load('../vectorizer/tfidf_vectorizer.pkl')

app = FastAPI()

class TextRequest(BaseModel):
    text: str

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

@app.post("/predict")
async def predict(text_request: TextRequest):
    text = text_request.text
    preprocessed_text = preprocess_text(text)
    tfidf_text = tfidf_vectorizer.transform([preprocessed_text])
    prediction = model.predict(tfidf_text)
    print(prediction)
    return {"label": f"{prediction}"}
