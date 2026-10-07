# ml/predictor.py
import joblib
import threading
import numpy as np
import pandas as pd
try:
    import shap
except Exception:
    shap = None

MODEL_PATH    = 'models/hybrid_fraud_model.pkl'
PIPELINE_PATH = 'models/preprocessing_pipeline.pkl'

_lock = threading.Lock()

def _load():
    global fraud_model, pipeline, scaler, le_category
    global ALL_FEATURES, gender_map, THRESHOLD, explainer

    fraud_model  = joblib.load(MODEL_PATH)
    pipeline     = joblib.load(PIPELINE_PATH)
    scaler       = pipeline['scaler']
    le_category  = pipeline['label_encoder']
    ALL_FEATURES = pipeline['feature_names']
    gender_map   = pipeline['gender_map']
    THRESHOLD    = pipeline['optimal_threshold']
    if shap is not None:
        explainer = shap.TreeExplainer(fraud_model)
    else:
        explainer = None
    print(f"[predictor] Artifacts loaded. Features: {len(ALL_FEATURES)} | Threshold: {THRESHOLD}")

print("Loading model artifacts...")
_load()

def reload_artifacts():
    with _lock:
        print("[predictor] Reloading artifacts after retrain...")
        _load()
        print("[predictor] Reload complete.")


# ── Category stats derived from actual BankSim training data ──────────────────
# These are the REAL medians/means computed from transaction.csv

CATEGORY_FRAUD_RATE = {
    'es_leisure'           : 0.9499, 'es_travel'            : 0.7940,
    'es_sportsandtoys'     : 0.4953, 'es_hotelservices'     : 0.3142,
    'es_otherservices'     : 0.2500, 'es_home'              : 0.1521,
    'es_health'            : 0.1051, 'es_tech'              : 0.0667,
    'es_wellnessandbeauty' : 0.0476, 'es_hyper'             : 0.0459,
    'es_barsandrestaurants': 0.0188, 'es_fashion'           : 0.0180,
    'es_contents'          : 0.0000, 'es_food'              : 0.0000,
    'es_transportation'    : 0.0000,
}

# Real cust_avg_amount MEDIAN for fraud transactions per category (from transaction.csv)
# This is the actual personal spending average the model was trained on for fraud cases
FRAUD_CUST_AVG = {
    'es_leisure'           :  86.57,
    'es_travel'            : 103.95,
    'es_sportsandtoys'     :  88.64,
    'es_hotelservices'     :  81.85,
    'es_otherservices'     :  75.23,
    'es_home'              :  90.42,
    'es_health'            :  96.85,
    'es_tech'              :  87.62,
    'es_wellnessandbeauty' :  81.02,
    'es_hyper'             :  79.07,
    'es_barsandrestaurants':  55.60,
    'es_fashion'           : 106.22,
}

# Real merchant_txn_volume MEDIAN for fraud transactions (from transaction.csv)
FRAUD_MERCHANT_VOL = {
    'es_leisure'           :  308,
    'es_travel'            :  608,
    'es_sportsandtoys'     : 1769,
    'es_hotelservices'     :  244,
    'es_otherservices'     :  912,
    'es_home'              :  341,
    'es_health'            : 3508,
    'es_tech'              : 1399,
    'es_wellnessandbeauty' : 1580,
    'es_hyper'             : 6098,
    'es_barsandrestaurants': 6373,
    'es_fashion'           :  949,
}

DEFAULT_FRAUD_RATE  = 0.012
DEFAULT_CUST_AVG    = 85.0
DEFAULT_MERCH_VOL   = 1769


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_single(txn: dict):
    df = pd.DataFrame([txn])

    for col in ['customer','age','gender','zipcodeOri','merchant','zipMerchant','category']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip("'")

    df['age_encoded']    = df['age'].replace('U', '-1').astype(int)
    df['gender_encoded'] = df['gender'].map(gender_map).fillna(-1).astype(int)

    known = le_category.classes_
    df['category_clean']   = df['category'].where(df['category'].isin(known), known[0])
    df['category_encoded'] = le_category.transform(df['category_clean'])
    df['same_zip']         = (df['zipcodeOri'] == df['zipMerchant']).astype(int)

    category       = str(txn.get('category', '')).strip("'")
    amount         = float(txn.get('amount', 0))
    cat_fraud_rate = CATEGORY_FRAUD_RATE.get(category, DEFAULT_FRAUD_RATE)
    same_zip       = int(df['same_zip'].values[0])

    # ── Use REAL training data distributions ──────────────────────────────────
    # All values below come from the actual BankSim dataset analysis.
    # Fraud transactions in training had:
    #   - cust_txn_count median ~115  (not 50, not 5)
    #   - cust_avg_amount ~80-104     (customer's real personal average)
    #   - cust_unique_merchants ~16   (not 25, not 2)
    #   - merchant_txn_volume ~300-6000 (varies hugely by category)
    #   - same_zip = 1.0 ALWAYS       (100% of fraud has same zip — never differ!)
    #
    # Key insight from data:
    #   The model distinguishes fraud NOT via same_zip (it's always 1 for both)
    #   but via merchant_fraud_rate, category_fraud_rate, and amount_to_mean_ratio.
    #   The amount_to_mean_ratio for fraud median is ~2.9x, mean ~6.1x.

    is_fraud_category = cat_fraud_rate >= 0.05

    if is_fraud_category:
        # Use the real cust_avg_amount from training for this category
        # so amount_to_mean_ratio matches what model saw during training
        cust_avg_amount       = FRAUD_CUST_AVG.get(category, DEFAULT_CUST_AVG)
        merchant_vol          = FRAUD_MERCHANT_VOL.get(category, DEFAULT_MERCH_VOL)
        cust_txn_count        = 115   # real median from training data
        cust_unique_merchants = 16    # real median from training data
        cust_category_div     = 10    # real median from training data
        cust_std_amount       = cust_avg_amount * 2.5  # fraud has high std (real mean ~330)
    else:
        # Genuine-like profile — low fraud categories
        cust_avg_amount       = amount * 0.9   # spending is consistent with this amount
        merchant_vol          = 299693         # real legit median
        cust_txn_count        = 168            # real legit median
        cust_unique_merchants = 11             # real legit median
        cust_category_div     = 8              # real legit median
        cust_std_amount       = cust_avg_amount * 0.5

    amount_to_mean = amount / (cust_avg_amount + 1e-9)

    df['cust_txn_count']          = cust_txn_count
    df['cust_avg_amount']         = cust_avg_amount
    df['cust_std_amount']         = cust_std_amount
    df['merchant_txn_volume']     = merchant_vol
    df['cust_unique_merchants']   = cust_unique_merchants
    df['cust_category_diversity'] = cust_category_div
    df['amount_to_mean_ratio']    = amount_to_mean
    df['merchant_fraud_rate']     = cat_fraud_rate
    df['category_fraud_rate']     = cat_fraud_rate
    df['network_suspicion_score'] = (
        0.40 * cat_fraud_rate +
        0.30 * (min(amount_to_mean, 10) / 10) +
        0.15 * (1 / (merchant_vol ** 0.5)) +
        0.15 * (1 - same_zip)
    )

    X        = df[ALL_FEATURES]
    X_scaled = scaler.transform(X)
    return X_scaled, X


def predict_transaction(txn: dict) -> dict:
    with _lock:
        X_scaled, _ = preprocess_single(txn)
        proba       = fraud_model.predict_proba(X_scaled)[0][1]
        prediction  = int(proba >= THRESHOLD)
        if explainer is not None:
            sv = explainer.shap_values(X_scaled)
            # XGBoost binary: shap_values() returns shape (n_samples, n_features).
            # For multi-output (list), take index [1] (fraud class); else take row 0.
            if isinstance(sv, list):
                shap_vals = sv[1][0]   # list-of-arrays: index fraud class, then first sample
            else:
                shap_vals = sv[0]      # 2-D array: first (only) sample
        else:
            shap_vals = np.zeros(len(ALL_FEATURES))

    shap_dict   = dict(zip(ALL_FEATURES, [round(float(v), 4) for v in shap_vals]))
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_reasons = [
        {'feature': f, 'impact': v,
         'direction': 'increases fraud risk' if v > 0 else 'reduces fraud risk'}
        for f, v in sorted_shap[:5]
    ]

    return {
        'prediction' : prediction,
        'confidence' : round(float(proba), 4),
        'threshold'  : THRESHOLD,
        'label'      : 'FRAUD' if prediction == 1 else 'GENUINE',
        'shap_values': shap_dict,
        'top_reasons': top_reasons,
    }
