# Agri-Demand Optimizer: Pilot Integration Guide

## 1. Executive Summary
This guide outlines the procedural integration of the **Agri-Demand Optimizer** into Kenyan agricultural supply chains (e.g., Farm to Feed operations). The goal is to maximize the diversion of surplus or "imperfect" produce from smallholder farmers to commercial kitchens by utilizing high-precision demand forecasting.

## 2. Strategic Objectives
*   **Post-Harvest Loss Reduction:** Predict demand with >0.90 AUC to move produce before it perishes.
*   **Farmer Income Stability:** Provide predictable volume requirements to ensure fair, consistent payments to rural producers.
*   **Operational Efficiency:** Optimize fleet logistics and cold-chain allocation based on 7-day and 14-day projections.

## 3. Implementation Workflow

### Phase A: Historical Alignment
*   Ensure the latest 12-week transaction history is loaded into the `data/` directory.
*   Run the **System Calibrator** (`pipeline/farm_recommender_elite_v11.py`) to align the model with current market regimes.

### Phase B: Pilot API Integration
*   Industrial procurement apps should point to the **Analytical Console API**:
    *   `POST /predict_demand`
    *   Inputs: `SKU`, `Customer Segment`, `Lag Quantity`.
*   Users should utilize the provided **Python/JS SDKs** for robust error handling.

### Phase C: Insight Review
*   Supply chain managers should review the **System Analytics Summary** before final logistics approval.
*   High-Confidence signals (>70%) should be prioritized for immediate procurement.

## 4. Business Value Metrics
*   **Target:** 15% reduction in stock-outs for critical SKUs.
*   **Target:** 20% increase in volume of "ugly" produce diverted to kitchens.
*   **Target:** 12% improvement in cold-chain energy efficiency via predictive logistics.

---
**System Engineering:** Christopher Okech | Nairobi, Kenya
