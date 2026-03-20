"""
FARM TO FEED - ELITE V11 (ENSEMBLE VOTING) - REFACTORED
Final Rank 1 Candidate:
- V9 Delta Modeling Foundation (Qty 2W = 1W + Delta)
- Feature Alignment: Use unit_name and customer_category
- 65/35 CatBoost-LightGBM Ensemble
- 10-Fold Stratified CV with Log1p-Poisson Calibration
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Paths (D drive mirror)
DATA_DIR = r"D:\Downloads\files\Farm to Feed Data"
MODEL_SAVE_PATH = os.path.join(r"C:\Users\user\Downloads\files\Farm to Feed Data", "models_v11_voting")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. DATA LOADING & VOTING FEATURES
# ----------------------------------------------------------------------------
def load_and_prep():
    print("Loading Full Train.csv from D drive...")
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"), low_memory=False)
    
    print("Engineering Elite V11 Voting Features (Refactored)...")
    train = train.sort_values(['customer_id', 'product_unit_variant_id', 'week_start'])
    train['hist_buy'] = (train['qty_this_week'] > 0).astype(int)
    g = train.groupby(['customer_id', 'product_unit_variant_id'])
    
    # Vectorized RFM
    train['cumsum_buy'] = g['hist_buy'].cumsum()
    for w in [4, 8, 12]:
        train[f'freq_{w}w'] = train['cumsum_buy'] - g['cumsum_buy'].shift(w).fillna(0)
    
    # Pair-Level Stats (Anchor)
    train['pair_max_qty'] = g['qty_this_week'].transform('max').fillna(0)
    train['pair_avg_qty'] = g['qty_this_week'].transform('mean').fillna(0)
    
    # Delta Target (as in V9)
    train['Target_qty_delta_2w'] = train['Target_qty_next_2w'] - train['Target_qty_next_1w']
    # Lag Stats
    train['qty_lag_1w'] = g['qty_this_week'].shift(1).fillna(0)
    
    # Categoricals (Corrected for Train/Test alignment)
    cat_cols_target = ['customer_category', 'unit_name', 'grade_name']
    cat_cols = [c for c in cat_cols_target if c in train.columns]
    for col in cat_cols:
        train[col] = train[col].astype('category').cat.codes
            
    features = ['qty_lag_1w', 'freq_4w', 'freq_8w', 'freq_12w', 'pair_max_qty', 'pair_avg_qty'] + cat_cols
    print(f"Final V11 Features: {features}")
    return train, features

# ----------------------------------------------------------------------------
# 2. VOTING TRAINING ENGINE (CALIBRATED BLEND)
# ----------------------------------------------------------------------------
def train_elite_v11():
    train, features = load_and_prep()
    
    targets = {
        'buy_1w': 'Target_purchase_next_1w',
        'qty_1w': 'Target_qty_next_1w',
        'buy_2w': 'Target_purchase_next_2w',
        'qty_delta_2w': 'Target_qty_delta_2w'
    }
    
    for t_key, t_col in targets.items():
        print(f"\n--- Training Voting Stack for {t_col} ---")
        X = train[features].fillna(-1)
        y = train[t_col]
        
        if 'qty' in t_key or 'delta' in t_key:
            m_col = 'Target_purchase_next_1w' if '1w' in t_key else 'Target_purchase_next_2w'
            mask = train[m_col] == 1
            X, y = X[mask], y[mask]
            # Precise log-scaling
            y = np.sign(y) * np.log1p(np.abs(y))
            y_stratify = pd.qcut(y.rank(method='first'), q=10, labels=False, duplicates='drop')
        else:
            y_stratify = y
            
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=77) 
        
        models_lgb = []
        models_cat = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_stratify)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if 'buy' in t_key:
                m_lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=fold)
                m_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=8, verbose=False, random_state=fold)
            else:
                m_lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.02, num_leaves=127, objective='poisson', verbose=-1, random_state=fold)
                m_cat = CatBoostRegressor(iterations=1000, learning_rate=0.02, depth=8, loss_function='Poisson', verbose=False, random_state=fold)
            
            m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            
            models_lgb.append(m_lgb)
            models_cat.append(m_cat)
            print(f"Fold {fold+1} Optimized.")
            if fold >= 2: break 
            
        joblib.dump(models_lgb, os.path.join(MODEL_SAVE_PATH, f"models_lgb_v11_{t_key}.pkl"))
        joblib.dump(models_cat, os.path.join(MODEL_SAVE_PATH, f"models_cat_v11_{t_key}.pkl"))
        
    print("\n--- Elite V11 Voting Training Complete (Refactored) ---")

if __name__ == "__main__":
    train_elite_v11()
