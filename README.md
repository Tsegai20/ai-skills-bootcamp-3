
# Day 3 — FastAPI ML Model Serving

## What I built
Production REST API serving a fine-tuned BERT model for news classification.

## Endpoints
- GET  /           — Health check
- POST /predict    — Single text classification
- POST /predict/batch — Batch classification up to 50 texts
- GET  /categories — Available categories
- GET  /docs       — Interactive Swagger documentation

## Performance
- Average response time: 846ms on CPU
- All predictions correct across 20 test requests
- Error handling with proper HTTP status codes

## Tech Stack
- FastAPI
- Uvicorn
- Pydantic
- HuggingFace Transformers
- PyTorch

## What I learned
- REST API design and HTTP methods
- Pydantic input validation
- Automatic API documentation with Swagger
- Error handling with HTTP status codes
- Batch processing endpoints
- Performance testing

## How to run
pip install -r requirements_day3.txt
uvicorn api:app --reload

## Author
Tsegai Yhdego
PhD Industrial Engineering — FAMU-FSU
AI/ML Researcher — R-SEAT Center
