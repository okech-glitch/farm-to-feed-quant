"""
FARM TO FEED - ELITE V11 SUBMISSION (ENSEMBLE VOTING) - REFACTORED
Final Rank 1 Lockdown Engine:
- Triple-Threshold Voting: Average [0.3, 0.4, 0.5] masks for smooth transition
- Feature Alignment: Use unit_name and customer_category
- Delta Re-assembly: Qty 2W = Qty 1W + Delta
- 65/35 CatBoost-LightGBM Precision Weighting
"""

import pandas as pd
import numpy as np
import os
import joblib

# Paths (D drive mirror)
DATA_DIR = r"D:\Downloads\files\Farm to Feed Data"
MODEL_PATH = r"C:\Users\user\Downloads\files\Farm to Feed Data\models_v11_voting"
SUBMISSION_PATH = r"C:\Users\user\Downloads\files\Farm to Feed Data\submissions"
os.makedirs(SUBMISSION_PATH, exist_ok=True)

def generate_elite_v11_submission():
    print("--- Generating Elite V11 Farm to Feed Submission (Voting) - Refactored ---")
    
    # 1. Load Data
    print("Loading datasets...")
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    
    # 2. Vectorized Feature Engineering (Refactored)
    print("Engineering Elite V11 Voting Features (Refactored)...")
    train = train.sort_values(['customer_id', 'product_unit_variant_id', 'week_start'])
    train['hist_buy'] = (train['qty_this_week'] > 0).astype(int)
    g = train.groupby(['customer_id', 'product_unit_variant_id'])
    
    # Pair-Specific Max/Avg (Anchors)
    pair_max = g['qty_this_week'].max().to_dict()
    pair_avg = g['qty_this_week'].mean().to_dict()
    
    latest_state = train.groupby(['customer_id', 'product_unit_variant_id']).tail(1).copy()
    latest_state['qty_lag_1w'] = latest_state['qty_this_week']
    
    for w in [4, 8, 12]:
        latest_state[f'freq_{w}w'] = g['hist_buy'].transform(lambda x: x.rolling(w, min_periods=1).sum()).iloc[latest_state.index]
    
    # Merge Features
    features = ['qty_lag_1w', 'freq_4w', 'freq_8w', 'freq_12w', 'pair_max_qty', 'pair_avg_qty', 'customer_category', 'unit_name', 'grade_name']
    test = test.merge(latest_state[['customer_id', 'product_unit_variant_id', 'qty_lag_1w', 'freq_4w', 'freq_8w', 'freq_12w']], 
                     on=['customer_id', 'product_unit_variant_id'], how='left').fillna(0)
    
    # Map Anchors & Categoricals
    test_keys_pair = list(zip(test.customer_id, test.product_unit_variant_id))
    test['pair_max_qty'] = [pair_max.get(k, 0) for k in test_keys_pair]
    test['pair_avg_qty'] = [pair_avg.get(k, 0) for k in test_keys_pair]
    
    # Ensure Categoricals are in Test
    if 'grade_name' not in test.columns:
        grade_map = train.groupby('product_unit_variant_id')['grade_name'].first().to_dict()
        test['grade_name'] = test['product_unit_variant_id'].map(grade_map).fillna('UNKNOWN')
    if 'unit_name' not in test.columns:
        unit_map = train.groupby('product_unit_variant_id')['unit_name'].first().to_dict()
        test['unit_name'] = test['product_unit_variant_id'].map(unit_map).fillna('UNKNOWN')
    if 'customer_category' not in test.columns:
        cust_map = train.groupby('customer_id')['customer_category'].first().to_dict()
        test['customer_category'] = test['customer_id'].map(cust_map).fillna('UNKNOWN')
    
    # Cat encode (must match train's codes if possible, or just re-encode consistently)
    for col in ['customer_category', 'unit_name', 'grade_name']:
        combined = pd.concat([train[col], test[col]]).astype('category')
        test[col] = combined[len(train):].cat.codes
    
    # 3. Voting Inference Loop
    targets = ['buy_1w', 'qty_1w', 'buy_2w', 'qty_delta_2w']
    results = {}
    X_test = test[features].fillna(-1)
    
    for t_key in targets:
        print(f"Running Inference for V11 {t_key} Voting Stack...")
        models_lgb = joblib.load(os.path.join(MODEL_PATH, f"models_lgb_v11_{t_key}.pkl"))
        models_cat = joblib.load(os.path.join(MODEL_PATH, f"models_cat_v11_{t_key}.pkl"))
        
        stack_preds = np.zeros(len(test))
        num_models = len(models_lgb)
        
        for ml, mc in zip(models_lgb, models_cat):
            if 'buy' in t_key:
                p_lgb = ml.predict_proba(X_test)[:, 1]
                p_cat = mc.predict_proba(X_test)[:, 1]
                stack_preds += (0.4 * p_lgb + 0.6 * p_cat) / num_models
            else:
                p_lgb = np.sign(ml.predict(X_test)) * np.expm1(np.abs(ml.predict(X_test)))
                p_cat = np.sign(mc.predict(X_test)) * np.expm1(np.abs(mc.predict(X_test)))
                stack_preds += (0.35 * p_lgb + 0.65 * p_cat) / num_models
        
        results[t_key] = stack_preds
        
    # 4. Final Triple-Threshold Assembly
    print("Assembling Final V11 Triple-Threshold Submission...")
    sub = pd.DataFrame({'ID': test['ID']})
    
    thresholds = [0.30, 0.40, 0.50]
    
    # Target 1: Purchase 1W
    sub['Target_purchase_next_1w'] = results['buy_1w']
    # Target 2: Qty 1W (Triple Voting)
    qty_1w_voted = np.zeros(len(test))
    for t in thresholds:
        qty_1w_voted += (results['buy_1w'] > t).astype(int) * results['qty_1w'] / len(thresholds)
    sub['Target_qty_next_1w'] = qty_1w_voted
    
    # Target 3: Purchase 2W
    sub['Target_purchase_next_2w'] = np.maximum(results['buy_2w'], sub['Target_purchase_next_1w'])
    # Target 4: Qty 2W (Delta + Triple Voting)
    qty_2w_base = sub['Target_qty_next_1w'] + results['qty_delta_2w']
    qty_2w_voted = np.zeros(len(test))
    for t in thresholds:
        qty_2w_voted += (results['buy_2w'] > t).astype(int) * qty_2w_base / len(thresholds)
    sub['Target_qty_next_2w'] = np.maximum(qty_2w_voted, sub['Target_qty_next_1w'])
    
    # Final Outlier Shield
    for col in ['Target_qty_next_1w', 'Target_qty_next_2w']:
        pair_clip = (test['pair_max_qty'] * 1.3).clip(1, 40)
        sub[col] = np.minimum(sub[col], pair_clip).clip(0, 50)
    
    save_file = os.path.join(SUBMISSION_PATH, "farm_to_feed_v11_ensemble_voting.csv")
    sub.to_csv(save_file, index=False)
    print(f"Elite V11 Voting Submission saved to: {save_file}")

if __name__ == "__main__":
    generate_elite_v11_submission()
