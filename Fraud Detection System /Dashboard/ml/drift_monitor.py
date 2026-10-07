# ml/drift_monitor.py
import numpy as np
import pandas as pd
import joblib
import shap
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

DRIFT_CHECK_EVERY   = 50
DRIFT_PSI_THRESHOLD = 0.2
BASELINE_FRAUD_RATE = 0.012


def compute_psi(baseline_rate: float, current_rate: float) -> float:
    eps      = 1e-6
    actual   = max(current_rate,  eps)
    expected = max(baseline_rate, eps)
    psi      = (actual - expected) * np.log(actual / expected)
    return round(float(psi), 6)


def check_drift(recent_transactions: list) -> dict:
    if len(recent_transactions) < DRIFT_CHECK_EVERY:
        return {'checked': False,
                'reason': f'Need at least {DRIFT_CHECK_EVERY} transactions',
                'drift_detected': False}

    predictions = [t['prediction'] for t in recent_transactions
                   if t['prediction'] is not None]
    if not predictions:
        return {'checked': False, 'reason': 'No predictions found', 'drift_detected': False}

    current_rate   = sum(predictions) / len(predictions)
    psi            = compute_psi(BASELINE_FRAUD_RATE, current_rate)
    drift_detected = psi > DRIFT_PSI_THRESHOLD

    return {
        'checked'       : True,
        'window_size'   : len(predictions),
        'fraud_rate'    : round(current_rate, 4),
        'baseline_rate' : BASELINE_FRAUD_RATE,
        'psi_score'     : psi,
        'drift_detected': drift_detected,
        'action_taken'  : 'retrain_triggered' if drift_detected else 'none',
    }


def retrain_model(transactions: list) -> dict:
    """
    Retrain on labelled transactions, save new .pkl files, then:
      1. Compute optimal threshold using Youden's J statistic and save to pipeline.
      2. Call predictor.reload_artifacts() so the live explainer
         immediately reflects the new model — no Flask restart needed.
      3. Compute a fresh global SHAP snapshot and save it to the DB
         so drift in feature importance is tracked over time.
    """
    from ml.predictor import ALL_FEATURES, pipeline

    print("[drift] Starting model retraining...")

    # ── FIX 1: use actual_label (not is_fraud_actual) ─────────────────────────
    df = pd.DataFrame(transactions)
    df = df[df['actual_label'].notna()].copy()

    if len(df) < 100:
        return {'success': False,
                'reason': f'Not enough labelled data ({len(df)} rows, need 100+)'}

    # ── Feature engineering (mirrors predictor.preprocess_single) ──────────────
    df['age_encoded']      = df['age'].replace('U', '-1').astype(int)
    gender_map             = pipeline['gender_map']
    df['gender_encoded']   = df['gender'].map(gender_map).fillna(-1).astype(int)

    le_category            = pipeline['label_encoder']
    known                  = le_category.classes_
    df['category_clean']   = df['category'].where(df['category'].isin(known), known[0])
    df['category_encoded'] = le_category.transform(df['category_clean'])

    from ml.predictor import CATEGORY_FRAUD_RATE, DEFAULT_FRAUD_RATE

    df['same_zip'] = (df['zipcodeOri'] == df['zipMerchant']).astype(int)

    # Use realistic per-category fraud rates (mirrors predictor.preprocess_single)
    df['category_fraud_rate']  = df['category'].map(
        lambda c: CATEGORY_FRAUD_RATE.get(str(c).strip("'"), DEFAULT_FRAUD_RATE)
    )
    df['merchant_fraud_rate']  = df['category_fraud_rate']

    is_fraud_cat = df['category_fraud_rate'] >= 0.05

    # Realistic behavioural features based on actual label and category
    df['cust_avg_amount']  = df['amount'] * 0.35           # amount is ~3x personal avg for fraud
    df['cust_std_amount']  = df['cust_avg_amount'] * 1.5
    df['cust_txn_count']   = np.where(is_fraud_cat, 115, 168)
    df['merchant_txn_volume']     = np.where(is_fraud_cat, 500, 50000)
    df['cust_unique_merchants']   = np.where(is_fraud_cat, 16, 11)
    df['cust_category_diversity'] = np.where(is_fraud_cat, 10, 8)
    df['amount_to_mean_ratio']    = df['amount'] / (df['cust_avg_amount'] + 1e-9)

    df['network_suspicion_score'] = (
        0.40 * df['merchant_fraud_rate'] +
        0.30 * (df['amount_to_mean_ratio'].clip(0, 10) / 10) +
        0.15 * (1 / (df['merchant_txn_volume'].clip(1, None) ** 0.5)) +
        0.15 * (1 - df['same_zip'])
    )

    X = df[ALL_FEATURES]
    # ── FIX 1: use actual_label (not is_fraud_actual) ─────────────────────────
    y = df['actual_label'].astype(int)

    new_scaler = RobustScaler()
    X_scaled   = new_scaler.fit_transform(X)

    try:
        sm           = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_scaled, y)
    except Exception:
        X_res, y_res = X_scaled, y

    new_model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    new_model.fit(X_res, y_res)

    # ── FIX 2: Compute optimal threshold using Youden's J statistic ───────────
    try:
        y_proba                  = new_model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, thresholds     = roc_curve(y, y_proba)
        optimal_idx              = np.argmax(tpr - fpr)          # Youden's J
        optimal_threshold        = float(thresholds[optimal_idx])
        optimal_threshold        = round(max(0.01, min(0.99, optimal_threshold)), 6)
        print(f"[drift] Optimal threshold computed: {optimal_threshold}")
    except Exception as e:
        # Fallback to old threshold if roc_curve fails
        optimal_threshold = pipeline.get('optimal_threshold', 0.5)
        print(f"[drift] WARNING: Threshold computation failed, keeping old value ({optimal_threshold}): {e}")

    version = datetime.now().strftime('v%Y%m%d_%H%M%S')
    joblib.dump(new_model, 'models/hybrid_fraud_model.pkl')

    updated_pipeline                    = pipeline.copy()
    updated_pipeline['scaler']          = new_scaler
    updated_pipeline['optimal_threshold'] = optimal_threshold   # FIX 2: save new threshold
    joblib.dump(updated_pipeline, 'models/preprocessing_pipeline.pkl')

    print(f"[drift] Retrain complete — version: {version} | threshold: {optimal_threshold}")

    # ── Step 1: Hot-reload predictor so live SHAP = new model ─────────────────
    try:
        from ml.predictor import reload_artifacts
        reload_artifacts()
        print("[drift] Live explainer updated to new model.")
    except Exception as e:
        print(f"[drift] WARNING: reload_artifacts() failed: {e}")

    # ── Step 2: Compute + store global SHAP snapshot for the new model ────────
    shap_snapshot = []
    try:
        from ml.predictor import explainer as new_explainer
        sample_X = X_scaled[:min(500, len(X_scaled))]
        sv = new_explainer.shap_values(sample_X)
        # XGBoost binary: sv may be a list [neg_class, pos_class] or plain 2D array
        if isinstance(sv, list):
            sv = sv[1]  # take the fraud (positive) class
        mean_abs = np.abs(sv).mean(axis=0)
        shap_snapshot = sorted(
            [{'feature': f, 'importance': round(float(v), 6)}
             for f, v in zip(ALL_FEATURES, mean_abs)],
            key=lambda x: x['importance'], reverse=True
        )
        print(f"[drift] SHAP snapshot computed ({len(shap_snapshot)} features).")
    except Exception as e:
        print(f"[drift] WARNING: SHAP snapshot failed: {e}")

    # Compute actual ROC-AUC on the original (non-resampled) data
    try:
        y_proba_eval     = new_model.predict_proba(X_scaled)[:, 1]
        computed_roc_auc = round(float(roc_auc_score(y, y_proba_eval)), 4)
        print(f"[drift] ROC-AUC: {computed_roc_auc}")
    except Exception as e2:
        print(f"[drift] WARNING: roc_auc_score failed: {e2}")
        computed_roc_auc = 0.0

    return {
        'success'           : True,
        'version'           : version,
        'samples_used'      : len(X_res),
        'roc_auc'           : computed_roc_auc,
        'trigger'           : 'drift_retrain',
        'optimal_threshold' : optimal_threshold,
        'shap_snapshot'     : shap_snapshot,
    }