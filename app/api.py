from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import FraudDetector

app = FastAPI()
detector = FraudDetector()

# Define the exact schema the API expects
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Hour: int

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # This ensures features are always in the correct order
        feature_dict = transaction.model_dump()
        features = [feature_dict[k] for k in feature_dict]
        
        score = detector.score(features)
        action = detector.get_action(score)
        
        return {
            "score": round(float(score), 4),
            "action": action,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))