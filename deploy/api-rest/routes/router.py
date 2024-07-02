from fastapi import APIRouter, HTTPException
from utils.tools import load_model, predict
from models.model import LoanPredictionRequest

router = APIRouter()

# Load the model during router startup
model_path = "resources/model.joblib"  # Replace with your actual cloud storage path
model = load_model(model_path)

# Endpoint for prediction
@router.post("/predict/")
def predict_loan_approval(request: LoanPredictionRequest):
    try:
        prediction = predict(model, request.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))