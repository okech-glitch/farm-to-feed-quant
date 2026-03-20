import requests
import json

class AgriDemandClient:
    """
    Client for the Agri-Demand Optimizer (Kenya Fresh Produce).
    Predicts shopping baskets to reduce food waste and stabilize farmer income.
    """
    def __init__(self, api_url="https://okechobonyo-farm-to-feed-elite.hf.space/predict_demand"):
        self.api_url = api_url

    def get_forecast(self, customer_category: str, sku_unit: str, lag_qty: float, freq_4w: int, peak_qty: float):
        """
        Request 7-day and 14-day demand projections for a specific SKU.
        """
        payload = {
            "customer_category": customer_category,
            "unit_name": sku_unit,
            "grade_name": "GRADE_01", # Default to standard pilot grade
            "qty_lag_1w": lag_qty,
            "freq_4w": freq_4w,
            "freq_8w": freq_4w * 2,  # Simulated estimation
            "freq_12w": freq_4w * 3, # Simulated estimation
            "pair_max_qty": peak_qty,
            "pair_avg_qty": (lag_qty + peak_qty) / 2
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}

# --- Usage Example ---
# client = AgriDemandClient()
# report = client.get_forecast("CUST_CAT_001", "UNIT_004", 12.5, 3, 25.0)
# print(f"7-Day Volume: {report['predictions']['week_1']['predicted_quantity']} units")
