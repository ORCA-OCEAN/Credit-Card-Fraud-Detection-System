# ml/generator.py
import random
import string

# ── Categories ────────────────────────────────────────────────────────────────
# From real BankSim data: these have high fraud rates AND enough fraud samples
FRAUD_CATEGORIES = [
    'es_sportsandtoys',  # 49.53% fraud — 1982 fraud cases (largest group)
    'es_health',         # 10.51% fraud — 1696 fraud cases (second largest)
    'es_leisure',        # 94.99% fraud — 474 cases
    'es_travel',         # 79.40% fraud — 578 cases
    'es_hotelservices',  # 31.42% fraud — 548 cases
    'es_home',           # 15.21% fraud — 302 cases
    'es_otherservices',  # 25.00% fraud — 228 cases
]

GENUINE_CATEGORIES = [
    'es_transportation', # 0.00% fraud — 505,119 cases
    'es_food',           # 0.00% fraud — 26,254 cases
    'es_contents',       # 0.00% fraud — 885 cases
    'es_barsandrestaurants', # 1.88%
    'es_hyper',          # 4.59%
]

GENDERS  = ['M', 'F']
AGES     = ['1', '2', '3', '4', '5', '6']
ZIPCODES = ['28007', '28008', '28009', '28010', '28011']

# Real category average amounts from training data
CATEGORY_AVG_AMOUNT = {
    'es_sportsandtoys' : 215.72,
    'es_health'        : 135.62,
    'es_leisure'       : 288.91,
    'es_travel'        : 2250.41,
    'es_hotelservices' : 205.61,
    'es_home'          : 165.67,
    'es_otherservices' : 135.88,
}

# Real cust_avg_amount median for fraud in each category (from transaction.csv analysis)
# Fraud transactions are 2.5-17x higher than the customer's personal average
FRAUD_CUST_AVG = {
    'es_sportsandtoys' :  88.64,
    'es_health'        :  96.85,
    'es_leisure'       :  86.57,
    'es_travel'        : 103.95,
    'es_hotelservices' :  81.85,
    'es_home'          :  90.42,
    'es_otherservices' :  75.23,
}


def _random_id(prefix, length=10):
    return prefix + ''.join(random.choices(string.digits, k=length))


def generate_genuine_transaction() -> dict:
    zipcode = random.choice(ZIPCODES)
    return {
        'step'       : random.randint(1, 179),
        'customer'   : _random_id('C'),
        'age'        : random.choice(AGES),
        'gender'     : random.choice(GENDERS),
        'zipcodeOri' : zipcode,
        'merchant'   : _random_id('M'),
        'zipMerchant': zipcode,
        'category'   : random.choice(GENUINE_CATEGORIES),
        'amount'     : round(random.uniform(5.0, 80.0), 2),
    }


def generate_fraud_transaction() -> dict:
    zipcode  = random.choice(ZIPCODES)
    category = random.choice(FRAUD_CATEGORIES)

    # Generate amount that is realistically 3-20x the customer's personal average
    # Real data: amount_to_mean_ratio median=2.9, mean=6.1
    # We use 3x-10x the real cust_avg to produce a believable high ratio
    cust_avg = FRAUD_CUST_AVG[category]
    # ratio between 3x and 10x the customer's normal spending
    ratio  = random.uniform(3.0, 10.0)
    amount = round(cust_avg * ratio, 2)

    # In real BankSim data same_zip is ALWAYS 1 for both fraud and legit
    # So we keep same zip — changing it does NOT help the model
    return {
        'step'       : random.randint(1, 179),
        'customer'   : _random_id('C'),
        'age'        : random.choice(AGES),
        'gender'     : random.choice(['F', 'F', 'F', 'M']),
        'zipcodeOri' : zipcode,
        'merchant'   : _random_id('M'),
        'zipMerchant': zipcode,
        'category'   : category,
        'amount'     : amount,
    }
