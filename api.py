from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Initialize app
app = FastAPI(
    title="News Classifier API",
    description="Fine-tuned BERT model for classifying news into World, Sports, Business, and Sci/Tech",
    version="1.0.0"
)

# Label names
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Load model on startup
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("./my_news_classifier")
model = AutoModelForSequenceClassification.from_pretrained("./my_news_classifier")
model.eval()
print("Model loaded successfully")

# Input schemas
class TextInput(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "NASA launches new rocket to explore Mars"
            }
        }

class BatchInput(BaseModel):
    texts: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "NASA launches new rocket to Mars",
                    "Manchester United beats Chelsea 3-1",
                    "Federal Reserve raises interest rates"
                ]
            }
        }

# Output schemas
class PredictionOutput(BaseModel):
    text: str
    predicted_category: str
    confidence: float
    all_probabilities: dict
    processing_time_ms: float

class BatchOutput(BaseModel):
    results: List[PredictionOutput]
    total_processed: int
    total_time_ms: float

# Helper function
def classify_text(text: str):
    start = time.time()
    
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "text": text,
        "predicted_category": LABEL_NAMES[predicted_class],
        "confidence": round(confidence, 4),
        "all_probabilities": {
            name: round(prob.item(), 4)
            for name, prob in zip(LABEL_NAMES, probabilities)
        },
        "processing_time_ms": round(elapsed, 2)
    }

# Endpoints
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "Fine-tuned BERT",
        "version": "1.0.0",
        "categories": LABEL_NAMES
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    if len(input.text) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Text too long. Maximum 1000 characters."
        )
    
    result = classify_text(input.text)
    return result

@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(input: BatchInput):
    if not input.texts:
        raise HTTPException(
            status_code=400,
            detail="Texts list cannot be empty"
        )
    
    if len(input.texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 texts per batch request"
        )
    
    start = time.time()
    results = [classify_text(text) for text in input.texts]
    total_time = (time.time() - start) * 1000
    
    return {
        "results": results,
        "total_processed": len(results),
        "total_time_ms": round(total_time, 2)
    }

@app.get("/categories")
def get_categories():
    return {
        "categories": LABEL_NAMES,
        "total": len(LABEL_NAMES),
        "description": {
            "World": "International news and politics",
            "Sports": "Sports news and events",
            "Business": "Business and financial news",
            "Sci/Tech": "Science and technology news"
        }
    }
