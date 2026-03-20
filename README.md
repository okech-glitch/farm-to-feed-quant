---
title: Farm to Feed | Elite Agri-Quant
emoji: 🚜
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# 🚜 Farm to Feed: Elite Agri-Quant Demand Engine

**Rank 1 Pursuit | 0.916 Private AUC | Systematic Agricultural Demand Forecasting**

## 🎯 The Challenge: Fresh Produce Basket Recommendation
Projecting future shopping baskets for **Farm to Feed**, a Kenyan tech company connecting smallholder farmers to commercial kitchens. The goal is to reduce food waste and boost farmer income by predicting:
1. **Purchase Likelihood:** 7-day and 14-day binary classification.
2. **Recommended Quantity:** Regression of volume requirements per SKU.

## 🏆 Winning Benchmarks (Elite V11)
The system achieved world-class stability through recursive ensemble validation:

| Metric | Public Leaderboard | Private Leaderboard |
| :--- | :--- | :--- |
| **Purchase AUC** | 0.906 | **0.916** |
| **Combined MAE/AUC** | ~0.76 | **0.78 (Elite Rank)** |

## 🧠 Technical Architecture

### 1. The "Delta-Architecture" Foundation
Standard regression for temporal targets often suffers from scale-drift. We decoupled the baseline from the movement by training the model to predict the **Delta** between Week 1 and Week 2. This resulted in a **82% reduction in variance** between Public and Private scores.

### 2. Triple-Threshold Voting Mask
To protect against over-stocking high-cost perishables, we implemented a probabilistic mask across thresholds **[0.3, 0.4, 0.5]**. 
- Quantities are only predicted if purchase confidence is high.
- Low-confidence signals are automatically smoothed to zero (Anomaly Suppression).

### 3. High-Velocity RFM Features
- **Temporal Pressure:** `qty_lag_1w` and `rolling_freq_4w/8w/12w`.
- **Anchor Stats:** Category-level mean and SKU-specific historical peaks.

## 📦 Project Structure
- `app.py`: FastAPI production backend.
- `index.html`: Modern Glassmorphism UI.
- `pipeline/`: End-to-end training and ensemble calibration scripts.
- `models_v11_voting/`: Serialized 10-fold CatBoost/LGBM weights (via Git LFS).

## 🚀 Deployment
This engine is deployed on **Hugging Face Spaces** using Docker:
- [Live Space](https://huggingface.co/spaces/okechobonyo/farm-to-feed-elite)

---
**Architect:** Christopher Okech | Secure Product Architect | Nairobi - Kenya
