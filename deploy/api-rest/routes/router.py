from fastapi import APIRouter, HTTPException

router = APIRouter()

# Load the model during router startup
model_path = "model.joblib"  # Replace with your actual cloud storage path
model = load_model(model_path)

# Endpoint for prediction
@app.post("/predict/")
def predict_loan_approval(request: LoanPredictionRequest):
    try:
        prediction = predict(model, request.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))