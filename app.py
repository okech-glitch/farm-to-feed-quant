import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import os

app = FastAPI(
    title="Farm to Feed Elite Scorer",
    description="Live Demand Forecasting for Rank 1 Zindi Architecture (V11 Voting)",
    version="1.1.0"
)

# ── Load Models ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "Farm to Feed Data" / "models_v11_voting"

if not MODEL_DIR.exists():
    # Fallback to local path if directory structure differs on HF
    MODEL_DIR = Path(__file__).parent / "models_v11_voting"

def load_ensemble(target):
    lgb = joblib.load(MODEL_DIR / f"models_lgb_v11_{target}.pkl")
    cat = joblib.load(MODEL_DIR / f"models_cat_v11_{target}.pkl")
    return lgb, cat

# Pre-load all 4 stacks
print("Loading Elite V11 Ensembles...")
ensembles = {
    'buy_1w': load_ensemble('buy_1w'),
    'qty_1w': load_ensemble('qty_1w'),
    'buy_2w': load_ensemble('buy_2w'),
    'qty_delta_2w': load_ensemble('qty_delta_2w')
}

# ── Pydantic Schema ─────────────────────────────────────────────────────────
class Scenario(BaseModel):
    customer_category: str = Field(..., example="CUST_CAT_001")
    unit_name: str = Field(..., example="UNIT_004")
    grade_name: str = Field(..., example="GRADE_01")
    qty_lag_1w: float = Field(..., example=5.0)
    freq_4w: int = Field(..., example=2)
    freq_8w: int = Field(..., example=4)
    freq_12w: int = Field(..., example=6)
    pair_max_qty: float = Field(..., example=20.0)
    pair_avg_qty: float = Field(..., example=8.5)

# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")

@app.post("/predict_demand")
def predict_demand(scenario: Scenario):
    try:
        # 1. Feature Prep (Aligned with V11 Training)
        data = {
            'qty_lag_1w': scenario.qty_lag_1w,
            'freq_4w': scenario.freq_4w,
            'freq_8w': scenario.freq_8w,
            'freq_12w': scenario.freq_12w,
            'pair_max_qty': scenario.pair_max_qty,
            'pair_avg_qty': scenario.pair_avg_qty,
            'customer_category': scenario.customer_category,
            'unit_name': scenario.unit_name,
            'grade_name': scenario.grade_name
        }
        df = pd.DataFrame([data])
        
        # Consistent Encoding (Simulated for this live API)
        # In a real HF space, these would be mapped to the training labels
        for col in ['customer_category', 'unit_name', 'grade_name']:
            # Mock hash-based encoding for demonstration or use saved encoders
            df[col] = df[col].apply(lambda x: abs(hash(str(x))) % 100)

        # 2. Triple-Threshold Voting Inference
        thresholds = [0.30, 0.40, 0.50]
        results = {}
        
        for t_key, (models_lgb, models_cat) in ensembles.items():
            preds = 0
            num_folds = len(models_lgb)
            for ml, mc in zip(models_lgb, models_cat):
                if 'buy' in t_key:
                    p_lgb = ml.predict_proba(df)[:, 1][0]
                    p_cat = mc.predict_proba(df)[:, 1][0]
                    preds += (0.4 * p_lgb + 0.6 * p_cat) / num_folds
                else:
                    p_lgb = np.expm1(np.abs(ml.predict(df)[0]))
                    p_cat = np.expm1(np.abs(mc.predict(df)[0]))
                    preds += (0.35 * p_lgb + 0.65 * p_cat) / num_folds
            results[t_key] = preds

        # 3. Triple Mask Voting Logic
        buy_1w = results['buy_1w']
        qty_1w_base = results['qty_1w']
        
        qty_1w_voted = 0
        for t in thresholds:
            qty_1w_voted += (qty_1w_base if buy_1w > t else 0) / len(thresholds)
            
        buy_2w = max(results['buy_2w'], buy_1w)
        qty_2w_raw = qty_1w_voted + results['qty_delta_2w']
        
        qty_2w_voted = 0
        for t in thresholds:
            qty_2w_voted += (qty_2w_raw if buy_2w > t else 0) / len(thresholds)
            
        # Clipping
        qty_1w_final = min(qty_1w_voted, scenario.pair_max_qty * 1.3)
        qty_2w_final = min(max(qty_2w_voted, qty_1w_final), scenario.pair_max_qty * 1.3)

        # 4. Architect Insights
        risk = "Elite" if buy_1w > 0.7 else "Mid-Tier" if buy_1w > 0.4 else "Low-Confidence"
        growth = "Positive" if results['qty_delta_2w'] > 0 else "Stable/Declining"
        
        insight = f"Strategic Anchor: {scenario.unit_name} shows a {risk} demand regime. "
        insight += f"The temporal delta is {growth}, indicating a {'growth' if growth=='Positive' else 'contraction'} period for this SKU."

        return {
            "status": "success",
            "predictions": {
                "week_1": {
                    "purchase_probability": round(buy_1w, 3),
                    "predicted_quantity": round(qty_1w_final, 2)
                },
                "week_2": {
                    "purchase_probability": round(buy_2w, 3),
                    "predicted_quantity": round(qty_2w_final, 2)
                }
            },
            "metrics": {
                "delta": round(results['qty_delta_2w'], 2),
                "confidence_regime": risk
            },
            "architect_insight": insight
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
