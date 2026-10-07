# app.py
from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from database import (
    init_db, insert_transaction, get_all_transactions,
    get_transaction_count, get_recent_transactions,
    insert_drift_log, get_drift_logs,
    insert_model_version, get_current_model_version, get_all_model_versions,
    insert_shap_snapshot, get_shap_snapshots, get_shap_snapshot_versions,
    get_transactions_by_model_version,
    set_actual_label, get_accuracy_stats,
)
from ml.predictor import predict_transaction
from ml.generator import generate_fraud_transaction, generate_genuine_transaction
from ml.drift_monitor import check_drift, retrain_model, DRIFT_CHECK_EVERY

app = Flask(__name__)

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── Generate transaction (auto) ────────────────────────────────────────────────
@app.route('/api/generate', methods=['POST'])
def generate():
    data     = request.get_json()
    txn_type = data.get('type', 'genuine')

    txn    = generate_fraud_transaction()   if txn_type == 'fraud'   else generate_genuine_transaction()
    source = 'auto_fraud'                   if txn_type == 'fraud'   else 'auto_genuine'

    result = predict_transaction(txn)
    mv     = get_current_model_version()

    row = {**txn, 'source': source,
           'prediction'  : result['prediction'],
           'confidence'  : result['confidence'],
           'shap_values' : json.dumps(result['shap_values']),
           'model_version': mv}

    tx_id = insert_transaction(row)
    _maybe_check_drift()

    return jsonify({'id': tx_id, 'transaction': txn, 'result': result})


# ── Manual transaction submit ──────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    txn = request.get_json()

    required = ['step','customer','age','gender','zipcodeOri',
                'merchant','zipMerchant','category','amount']
    for field in required:
        if field not in txn:
            return jsonify({'error': f'Missing field: {field}'}), 400

    txn['amount'] = float(txn['amount'])
    txn['step']   = int(txn['step'])

    result = predict_transaction(txn)
    mv     = get_current_model_version()

    row = {**txn, 'source': 'manual',
           'prediction'  : result['prediction'],
           'confidence'  : result['confidence'],
           'shap_values' : json.dumps(result['shap_values']),
           'model_version': mv}

    tx_id = insert_transaction(row)
    _maybe_check_drift()

    return jsonify({'id': tx_id, 'transaction': txn, 'result': result})


# ── Transactions ───────────────────────────────────────────────────────────────
@app.route('/api/transactions', methods=['GET'])
def transactions():
    limit = request.args.get('limit', 100, type=int)
    return jsonify(get_all_transactions(limit))


# ── Dashboard stats ────────────────────────────────────────────────────────────
@app.route('/api/stats', methods=['GET'])
def stats():
    rows        = get_all_transactions(limit=1000)
    total       = len(rows)
    fraud_count = sum(1 for r in rows if r['prediction'] == 1)
    genuine     = total - fraud_count
    valid_conf  = [r['confidence'] for r in rows if r.get('confidence') is not None]
    avg_conf    = round(sum(valid_conf) / len(valid_conf), 4) if valid_conf else 0

    by_source = {}
    for r in rows:
        s = r['source'] or 'unknown'
        by_source[s] = by_source.get(s, 0) + 1

    current_version = get_current_model_version()

    return jsonify({
        'total'          : total,
        'fraud'          : fraud_count,
        'genuine'        : genuine,
        'fraud_rate'     : round(fraud_count / total, 4) if total else 0,
        'avg_conf'       : avg_conf,
        'by_source'      : by_source,
        'current_version': current_version,
    })


# ── Model versions ─────────────────────────────────────────────────────────────
@app.route('/api/model_versions', methods=['GET'])
def model_versions():
    return jsonify(get_all_model_versions())


# ── Drift logs ─────────────────────────────────────────────────────────────────
@app.route('/api/drift', methods=['GET'])
def drift():
    return jsonify(get_drift_logs(limit=20))


# ── Manual drift check ─────────────────────────────────────────────────────────
@app.route('/api/drift/check', methods=['POST'])
def drift_check():
    recent = get_recent_transactions(limit=200)
    result = check_drift(recent)

    if result.get('checked'):
        insert_drift_log({
            'window_size'   : result['window_size'],
            'fraud_rate'    : result['fraud_rate'],
            'baseline_rate' : result['baseline_rate'],
            'psi_score'     : result['psi_score'],
            'drift_detected': int(result['drift_detected']),
            'action_taken'  : result['action_taken'],
        })

        if result['drift_detected']:
            all_txns       = get_recent_transactions(limit=5000)
            retrain_result = retrain_model(all_txns)

            if retrain_result['success']:
                insert_model_version({
                    'version': retrain_result['version'],
                    'roc_auc': retrain_result['roc_auc'],
                    'trigger': 'drift_retrain',
                    'notes'  : f"Auto retrain. Samples: {retrain_result['samples_used']}",
                })
                # Persist SHAP snapshot so drift in feature importance is trackable
                if retrain_result.get('shap_snapshot'):
                    insert_shap_snapshot(retrain_result['version'],
                                         retrain_result['shap_snapshot'])

            result['retrain'] = retrain_result

    return jsonify(result)


# ── SHAP: Global feature importance ───────────────────────────────────────────
@app.route('/api/shap/global', methods=['GET'])
def shap_global():
    """
    Compute mean |SHAP| across transactions made by the CURRENT model version.
    This ensures the chart is never polluted by SHAP values from a stale model.
    Falls back to a pre-computed snapshot if available.
    """
    from ml.predictor import ALL_FEATURES, explainer, preprocess_single

    current_version = get_current_model_version()

    # Try pre-computed snapshot first (fast path — post retrain)
    snapshot = get_shap_snapshots(version=current_version)
    if snapshot:
        return jsonify({
            'features'      : [{'feature': r['feature_name'], 'importance': r['mean_abs_shap']}
                                for r in snapshot],
            'sample_size'   : '(snapshot)',
            'model_version' : current_version,
            'source'        : 'snapshot',
        })

    # Slow path — compute live from stored SHAP values for current version
    rows = get_transactions_by_model_version(current_version, limit=300)
    if not rows:
        # Fallback: use all transactions if version column was just added
        rows = get_recent_transactions(limit=200)

    if not rows:
        return jsonify({'error': 'No transactions yet'}), 404

    accumulator = None
    count       = 0
    for row in rows:
        stored = row.get('shap_values')
        if not stored:
            continue
        try:
            shap_dict = json.loads(stored)
            vals = np.array([abs(shap_dict.get(f, 0.0)) for f in ALL_FEATURES])
            accumulator = vals if accumulator is None else accumulator + vals
            count += 1
        except Exception:
            continue

    if count == 0 or accumulator is None:
        return jsonify({'error': 'Could not aggregate SHAP values'}), 500

    mean_abs = (accumulator / count).tolist()
    features = sorted(
        [{'feature': f, 'importance': round(v, 4)} for f, v in zip(ALL_FEATURES, mean_abs)],
        key=lambda x: x['importance'], reverse=True
    )
    return jsonify({
        'features'      : features,
        'sample_size'   : count,
        'model_version' : current_version,
        'source'        : 'live',
    })


# ── SHAP: Per-transaction waterfall ───────────────────────────────────────────
@app.route('/api/shap/explain/<int:tx_id>', methods=['GET'])
def shap_explain(tx_id):
    from database import get_connection
    _conn = get_connection()
    _cur  = _conn.cursor()
    _cur.execute('SELECT * FROM transactions WHERE id = ?', (tx_id,))
    _r    = _cur.fetchone()
    _conn.close()
    row = dict(_r) if _r else None
    if not row:
        return jsonify({'error': 'Transaction not found'}), 404

    stored = row.get('shap_values')
    if not stored:
        return jsonify({'error': 'No SHAP data for this transaction'}), 404

    FRAUD_REASON_TEMPLATES = {
        'amount'                  : 'Unusually high transaction amount',
        'amount_to_mean_ratio'    : 'Amount significantly above the customer normal spending',
        'merchant_fraud_rate'     : 'High fraud rate at this merchant',
        'network_suspicion_score' : 'High network-level suspicion score',
        'category_fraud_rate'     : 'Transaction in a high-risk merchant category',
        'cust_txn_count'          : 'Abnormal transaction frequency for this customer',
        'cust_std_amount'         : 'High volatility in spending pattern',
        'same_zip'                : 'Transaction outside customer home area',
        'merchant_txn_volume'     : 'Low-volume merchant with suspicious activity',
        'cust_unique_merchants'   : 'Unusually high number of merchants (potential card testing)',
    }
    FEATURE_DESCRIPTIONS = {
        'amount'                  : 'Transaction amount',
        'age_encoded'             : 'Cardholder age group',
        'gender_encoded'          : 'Cardholder gender',
        'category_encoded'        : 'Merchant category',
        'same_zip'                : 'Customer and merchant are in the same area',
        'step'                    : 'Time of transaction',
        'cust_txn_count'          : 'Customer transaction frequency',
        'cust_avg_amount'         : 'Customer average spending',
        'cust_std_amount'         : 'Customer spending variability',
        'amount_to_mean_ratio'    : 'Transaction amount relative to personal average',
        'merchant_fraud_rate'     : 'Merchant historical fraud rate',
        'merchant_txn_volume'     : 'Merchant transaction volume',
        'cust_unique_merchants'   : 'Number of different merchants visited',
        'network_suspicion_score' : 'Overall network suspicion score',
        'category_fraud_rate'     : 'Fraud rate in this merchant category',
        'cust_category_diversity' : 'Customer shopping category diversity',
    }

    try:
        shap_dict = json.loads(stored)
        items     = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        waterfall = [{'feature': k, 'value': v} for k, v in items]
        reasons   = []
        for feat, val in items[:5]:
            if val > 0.005:
                desc = FRAUD_REASON_TEMPLATES.get(
                    feat,
                    FEATURE_DESCRIPTIONS.get(feat, feat) + ' contributed to risk'
                )
                reasons.append({'text': desc, 'shap': round(val, 4)})

        return jsonify({
            'tx_id'        : tx_id,
            'waterfall'    : waterfall,
            'reasons'      : reasons,
            'prediction'   : row.get('prediction'),
            'confidence'   : row.get('confidence'),
            'model_version': row.get('model_version', 'v_original'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── SHAP: Drift comparison across model versions ───────────────────────────────
@app.route('/api/shap/drift_comparison', methods=['GET'])
def shap_drift_comparison():
    """
    Return SHAP snapshots for ALL model versions so the dashboard
    can visualise how feature importance shifted after each retrain.
    """
    versions  = get_shap_snapshot_versions()
    all_snaps = get_shap_snapshots()   # all versions

    # Group by version
    by_version = {}
    for row in all_snaps:
        v = row['model_version']
        if v not in by_version:
            by_version[v] = []
        by_version[v].append({'feature': row['feature_name'],
                               'importance': row['mean_abs_shap']})

    return jsonify({
        'versions'   : [v['model_version'] for v in versions],
        'by_version' : by_version,
    })


# ── Feedback: mark actual label ───────────────────────────────────────────────
@app.route('/api/feedback/<int:tx_id>', methods=['POST'])
def feedback(tx_id):
    data         = request.get_json()
    actual_label = data.get('actual_label')
    if actual_label not in [0, 1]:
        return jsonify({'error': 'actual_label must be 0 or 1'}), 400
    set_actual_label(tx_id, actual_label)
    return jsonify({'success': True, 'tx_id': tx_id, 'actual_label': actual_label})


# ── Debug accuracy raw data ────────────────────────────────────────────────────
@app.route('/api/accuracy/debug', methods=['GET'])
def accuracy_debug():
    from database import get_connection
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, prediction, actual_label,
               CASE WHEN prediction=1 AND actual_label=1 THEN 'TP'
                    WHEN prediction=0 AND actual_label=0 THEN 'TN'
                    WHEN prediction=1 AND actual_label=0 THEN 'FP'
                    WHEN prediction=0 AND actual_label=1 THEN 'FN'
                    ELSE 'unreviewed' END as result_type
        FROM transactions
        WHERE actual_label IS NOT NULL
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify(rows)


# ── Accuracy stats from reviewed transactions ──────────────────────────────────
@app.route('/api/accuracy', methods=['GET'])
def accuracy():
    stats = get_accuracy_stats()
    total = stats['total_reviewed'] or 0
    correct = stats['correct'] or 0
    fp  = stats['false_positives'] or 0
    fn  = stats['false_negatives'] or 0
    tp  = stats['true_positives']  or 0
    tn  = stats['true_negatives']  or 0

    accuracy_pct = round(correct / total * 100, 1) if total else None
    precision    = round(tp / (tp + fp) * 100, 1)  if (tp + fp) > 0 else None
    recall       = round(tp / (tp + fn) * 100, 1)  if (tp + fn) > 0 else None

    return jsonify({
        'total_reviewed' : total,
        'correct'        : correct,
        'accuracy'       : accuracy_pct,
        'precision'      : precision,
        'recall'         : recall,
        'true_positives' : tp,
        'true_negatives' : tn,
        'false_positives': fp,
        'false_negatives': fn,
    })


# ── Internal: auto drift check ─────────────────────────────────────────────────
def _maybe_check_drift():
    count = get_transaction_count()
    if count > 0 and count % DRIFT_CHECK_EVERY == 0:
        recent = get_recent_transactions(limit=200)
        result = check_drift(recent)
        if result.get('checked'):
            insert_drift_log({
                'window_size'   : result['window_size'],
                'fraud_rate'    : result['fraud_rate'],
                'baseline_rate' : result['baseline_rate'],
                'psi_score'     : result['psi_score'],
                'drift_detected': int(result['drift_detected']),
                'action_taken'  : result['action_taken'],
            })
            if result['drift_detected']:
                all_txns       = get_recent_transactions(limit=5000)
                retrain_result = retrain_model(all_txns)
                if retrain_result['success']:
                    insert_model_version({
                        'version': retrain_result['version'],
                        'roc_auc': retrain_result['roc_auc'],
                        'trigger': 'auto_drift',
                        'notes'  : f"Auto retrain. Samples: {retrain_result['samples_used']}",
                    })
                    if retrain_result.get('shap_snapshot'):
                        insert_shap_snapshot(retrain_result['version'],
                                             retrain_result['shap_snapshot'])


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)