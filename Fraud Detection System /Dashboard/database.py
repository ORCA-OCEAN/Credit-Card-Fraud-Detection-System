# database.py
import sqlite3
from datetime import datetime

DB_PATH = 'fraud.db'

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT    DEFAULT (datetime('now')),
            step            INTEGER,
            customer        TEXT,
            age             TEXT,
            gender          TEXT,
            zipcodeOri      TEXT,
            merchant        TEXT,
            zipMerchant     TEXT,
            category        TEXT,
            amount          REAL,
            source          TEXT,
            prediction      INTEGER,
            confidence      REAL,
            shap_values     TEXT,
            model_version   TEXT    DEFAULT 'v_original',
            actual_label    INTEGER DEFAULT NULL
        )
    ''')

    # Upgrade existing DBs that lack model_version column
    try:
        cursor.execute("ALTER TABLE transactions ADD COLUMN model_version TEXT DEFAULT 'v_original'")
        conn.commit()
    except Exception:
        pass

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            version    TEXT,
            roc_auc    REAL,
            trigger    TEXT,
            notes      TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drift_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            checked_at     TEXT DEFAULT (datetime('now')),
            window_size    INTEGER,
            fraud_rate     REAL,
            baseline_rate  REAL,
            psi_score      REAL,
            drift_detected INTEGER,
            action_taken   TEXT
        )
    ''')

    # New: stores global SHAP importance snapshot after every retrain
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shap_snapshots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at   TEXT DEFAULT (datetime('now')),
            model_version TEXT,
            feature_name  TEXT,
            mean_abs_shap REAL
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized.")

# ── Transactions ───────────────────────────────────────────────────────────────

def insert_transaction(data: dict) -> int:
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO transactions
            (step,customer,age,gender,zipcodeOri,merchant,
             zipMerchant,category,amount,source,
             prediction,confidence,shap_values,model_version)
        VALUES
            (:step,:customer,:age,:gender,:zipcodeOri,:merchant,
             :zipMerchant,:category,:amount,:source,
             :prediction,:confidence,:shap_values,:model_version)
    ''', data)
    conn.commit()
    tx_id = cursor.lastrowid
    conn.close()
    return tx_id

def get_all_transactions(limit=100):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transactions ORDER BY id DESC LIMIT ?', (limit,))
    rows   = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_transaction_count():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as cnt FROM transactions')
    count  = cursor.fetchone()['cnt']
    conn.close()
    return count

def get_recent_transactions(limit=200):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transactions ORDER BY id DESC LIMIT ?', (limit,))
    rows   = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_transactions_by_model_version(version: str, limit=500):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM transactions WHERE model_version=? ORDER BY id DESC LIMIT ?',
        (version, limit))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

# ── Drift log ──────────────────────────────────────────────────────────────────

def insert_drift_log(data: dict):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO drift_log
            (window_size,fraud_rate,baseline_rate,psi_score,drift_detected,action_taken)
        VALUES
            (:window_size,:fraud_rate,:baseline_rate,:psi_score,:drift_detected,:action_taken)
    ''', data)
    conn.commit()
    conn.close()

def get_drift_logs(limit=20):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM drift_log ORDER BY id DESC LIMIT ?', (limit,))
    rows   = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

# ── Model versions ────────────────────────────────────────────────────────────

def insert_model_version(data: dict):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO model_versions (version,roc_auc,trigger,notes) VALUES (:version,:roc_auc,:trigger,:notes)',
        data)
    conn.commit()
    conn.close()

def get_current_model_version() -> str:
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT version FROM model_versions ORDER BY id DESC LIMIT 1')
    row    = cursor.fetchone()
    conn.close()
    return row['version'] if row else 'v_original'

def get_all_model_versions():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM model_versions ORDER BY id DESC')
    rows   = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

# ── SHAP snapshots ────────────────────────────────────────────────────────────

def insert_shap_snapshot(version: str, feature_importances: list):
    """Save per-feature global SHAP importance after every retrain."""
    conn   = get_connection()
    cursor = conn.cursor()
    for item in feature_importances:
        cursor.execute(
            'INSERT INTO shap_snapshots (model_version,feature_name,mean_abs_shap) VALUES (?,?,?)',
            (version, item['feature'], item['importance']))
    conn.commit()
    conn.close()

def get_shap_snapshots(version: str = None):
    conn   = get_connection()
    cursor = conn.cursor()
    if version:
        cursor.execute(
            'SELECT * FROM shap_snapshots WHERE model_version=? ORDER BY mean_abs_shap DESC',
            (version,))
    else:
        cursor.execute('SELECT * FROM shap_snapshots ORDER BY captured_at DESC, mean_abs_shap DESC')
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_shap_snapshot_versions():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT model_version, MIN(captured_at) as captured_at
        FROM shap_snapshots GROUP BY model_version ORDER BY captured_at DESC
    ''')
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

if __name__ == '__main__':
    init_db()

# ── Feedback / Ground Truth ────────────────────────────────────────────────────

def set_actual_label(tx_id: int, actual_label: int):
    """Save human feedback: 0=genuine, 1=fraud."""
    conn   = get_connection()
    cursor = conn.cursor()
    # Add column if it doesn't exist yet (upgrade old DBs)
    try:
        cursor.execute("ALTER TABLE transactions ADD COLUMN actual_label INTEGER DEFAULT NULL")
        conn.commit()
    except Exception:
        pass
    cursor.execute(
        "UPDATE transactions SET actual_label = ? WHERE id = ?",
        (actual_label, tx_id)
    )
    conn.commit()
    conn.close()

def get_accuracy_stats():
    """Compute model accuracy from reviewed (labelled) transactions."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*)                                                         AS total_reviewed,
            SUM(CASE WHEN prediction = actual_label THEN 1 ELSE 0 END)      AS correct,
            SUM(CASE WHEN prediction=1 AND actual_label=0 THEN 1 ELSE 0 END) AS false_positives,
            SUM(CASE WHEN prediction=0 AND actual_label=1 THEN 1 ELSE 0 END) AS false_negatives,
            SUM(CASE WHEN prediction=1 AND actual_label=1 THEN 1 ELSE 0 END) AS true_positives,
            SUM(CASE WHEN prediction=0 AND actual_label=0 THEN 1 ELSE 0 END) AS true_negatives
        FROM transactions
        WHERE actual_label IS NOT NULL
    """)
    row = dict(cursor.fetchone())
    conn.close()
    return row