import uvicorn
import joblib
import re
import string
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spam Email Classifier")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and vectorizer
MODEL_PATH = "model/classifier.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model/vectorizer: {e}")
        # We don't raise here to allow the app to start, but predictions will fail
        pass

# Preprocessing function (must match training)
def preprocess_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Handle Unicode (basic cleaning if needed, though Python strings are unicode)
    # Removing non-printable characters or weird symbols if necessary
    text = text.encode('ascii', 'ignore').decode('ascii') 
    return text

class EmailRequest(BaseModel):
    content: str

@app.on_event("startup")
async def startup_event():
    load_artifacts()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_spam(email: EmailRequest):
    if not model or not vectorizer:
        # Try loading again (maybe trained after startup)
        load_artifacts()
        if not model or not vectorizer:
            raise HTTPException(status_code=500, detail="Model not trained yet.")

    processed_text = preprocess_text(email.content)
    
    # Vectorize
    features = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(features)[0] # 0 = Ham, 1 = Spam (usually)
    probability = model.predict_proba(features)[0]
    
    # Map prediction to label
    label = "Spam" if prediction == 1 else "Ham"
    confidence = float(probability[1]) if prediction == 1 else float(probability[0])
    
    return {
        "result": label,
        "confidence": f"{confidence * 100:.2f}%",
        "is_spam": bool(prediction == 1)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
