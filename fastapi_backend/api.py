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

# Exception handling
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"message": "Validation Error", "errors": exc.errors()})

# Predict
@app.post("/predict")
async def predict(text_request: TextRequest):
    text = text_request.text
    preprocessed_text = preprocess_text(text)
    tfidf_text = tfidf_vectorizer.transform([preprocessed_text])
    prediction = model.predict(tfidf_text)
    print(prediction)
    return {"label": f"{prediction}"}


# Documentation
@app.get("/", response_model=List[str])
async def read_root():
    """
    Welcome to the Fake News Detection API!
    
    You can make predictions by sending POST requests to /predict with JSON payload containing 'text'.
    The API will return the predicted label for the provided text.
    """
    return []
