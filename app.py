import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Path to the model and tokenizer directory (ensure these are the correct paths)
model_dir = r'C:\Users\harsh\OneDrive\Desktop\Final_Project\BERT\saved_model'

# Load the tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    raise Exception(f"Error loading model or tokenizer: {e}")

# Check if CUDA (GPU) is available and move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize FastAPI app
app = FastAPI()

# Define a request model for receiving the review text
class ReviewText(BaseModel):
    text: str

# Define the prediction endpoint
@app.post("/predict")
async def predict(review: ReviewText):
    try:
        # Tokenize the input text
        inputs = tokenizer(
            review.text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )
        
        # Move tensors to the correct device (CPU or GPU)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # Map predicted class to label (you should update this based on your class labels)
        label_map = {0: "CG", 1: "OR"}  # Adjust this mapping to match your label encoding
        label = label_map.get(predicted_class, "Unknown")

        return {"label": label, "class_id": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Optional: Add a root endpoint for basic health check
@app.get("/")
async def read_root():
    return {"message": "Model is ready to make predictions!"}
