from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os

app = FastAPI()

# ✅ Enable CORS for UI Connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_pipeline = None
default_threshold = 0.07  # Default threshold from your notebook 04 tuning

# 🔥 LOAD THE PIPELINE BUNDLE FROM YOUR MODELS FOLDER
try:
    path = None
    possible_paths = ["models/fraud_xgb.joblib", "../models/fraud_xgb.joblib", "fraud_xgb.joblib"]
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
            
    if path:
        bundle = joblib.load(path)
        if isinstance(bundle, dict):
            model_pipeline = bundle.get("model")
            default_threshold = bundle.get("threshold", 0.07)
        else:
            model_pipeline = bundle
        print(f"✅ SUCCESS: Complete ColumnTransformer Pipeline Loaded from '{path}'!")
        print(f"🎯 Threshold Configured: {default_threshold}")
    else:
        print("❌ ERROR: fraud_xgb.joblib file not found anywhere!")
except Exception as e:
    print(f"❌ ERROR During Loading: {e}")

# Frontend Payload Schema
class TransactionInput(BaseModel):
    amt: float
    city_pop: int
    is_male: int
    age: int
    hour: int
    day_of_week: int
    month: int
    trans_count_24h: int
    category_entertainment: int
    category_food_dining: int
    category_gas_transport: int
    category_grocery_net: int
    category_grocery_pos: int
    category_health_fitness: int
    category_home: int
    category_kids_pets: int
    category_misc_net: int
    category_misc_pos: int
    category_personal_care: int
    category_shopping_net: int
    category_shopping_pos: int
    category_travel: int

@app.post("/predict")
def predict_fraud(data: TransactionInput):
    if model_pipeline is None:
        return {"status": "Error", "error": "Model Pipeline not initialized on server layer."}
        
    try:
        input_dict = data.model_dump()
        
        # 🔍 Mapping category flags back to the original categorical string
        category_mapping = {
            'category_entertainment': 'entertainment', 'category_food_dining': 'food_dining',
            'category_gas_transport': 'gas_transport', 'category_grocery_net': 'grocery_net',
            'category_grocery_pos': 'grocery_pos', 'category_health_fitness': 'health_fitness',
            'category_home': 'home', 'category_kids_pets': 'kids_pets',
            'category_misc_net': 'misc_net', 'category_misc_pos': 'misc_pos',
            'category_personal_care': 'personal_care', 'category_shopping_net': 'shopping_net',
            'category_shopping_pos': 'shopping_pos', 'category_travel': 'travel'
        }
        
        resolved_category = 'grocery_pos'
        for key, text_val in category_mapping.items():
            if input_dict.get(key, 0) == 1:
                resolved_category = text_val
                break

        # 🎯 Reconstructing the exact dataset structure including the missing transformer tokens
        raw_row = {
            'amount': float(input_dict['amt']),
            'city_pop': int(input_dict['city_pop']),
            'is_male': int(input_dict['is_male']),
            'age': int(input_dict['age']),
            'hour': int(input_dict['hour']),
            'day_of_week': int(input_dict['day_of_week']),
            'month': int(input_dict['month']),
            'prev_24h_tx_count_card': int(input_dict['trans_count_24h']),
            'category': str(resolved_category),
            
            # 🔥 FIX: Adding the 6 strictly demanded columns by your ColumnTransformer Pipeline
            'merchant_cat': str(resolved_category),
            'merchant_cat_rare': 0,
            'country': 'US',
            'dayofweek': int(input_dict['day_of_week']),
            'city': 'unknown',
            'device_type': 'mobile',
            
            # Dynamic dynamic features mirroring your training notebooks calculations
            'log_amount': np.log1p(float(input_dict['amt'])),
            'velocity_ratio': 1.0,
            'avg_tx_amt_24h': float(input_dict['amt']),
            'amount_vs_24h_zscore': 0.0,
            'prev_1h_tx_count_card': 0.0,
            'prev_24h_amt_card': float(input_dict['amt']),
            'velocity_amt_1h': 0.0,
            'is_international': 0,
            'is_night': 1 if int(input_dict['hour']) < 6 or int(input_dict['hour']) > 22 else 0,
            'high_velocity_flag': 0,
            'is_weekend': 1 if int(input_dict['day_of_week']) >= 5 else 0,
            'channel': 'mobile'
        }
        
        # Build clean dataframe matching pipeline training features
        df_inference = pd.DataFrame([raw_row])
        
        # Run final inference score safely
        risk_probability = float(model_pipeline.predict_proba(df_inference)[0, 1])
        is_fraudulent = bool(risk_probability >= default_threshold)

        return {
            "status": "Success",
            "risk_prob": risk_probability,
            "is_fraud": is_fraudulent,
            "threshold_used": default_threshold
        }
        
    except Exception as e:
        print("API Execution Warning Exception:", str(e))
        return {"status": "Error", "error": f"ColumnTransformer Inference Block Failure: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)