import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ========== 1. LOAD YOUR EXACT DATA ==========
df = pd.read_excel("Task 3 and 4_Loan_Data.xlsx") 
# If you saved as CSV use: df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

print("Data shape:", df.shape)
print(df.head(3))

# Drop ID, use 'default' as target
X = df.drop(columns=['customer_id', 'default']) 
y = df['default']

# ========== 2. TRAIN PD MODEL + COMPARE ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
}

best_model = None
best_auc = 0
use_scaler = False

print("\n--- Model Comparison ---")
for name, model in models.items():
    if name == "LogisticRegression":
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        temp_scaler = True
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        temp_scaler = False
        
    auc = roc_auc_score(y_test, proba)
    print(f"{name} AUC: {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_model = model
        use_scaler = temp_scaler

print(f"\n>>> Best Model: {type(best_model).__name__} | AUC = {best_auc:.4f} <<<")

# ========== 3. EXPECTED LOSS FUNCTION ==========
RECOVERY_RATE = 0.10  # 10% per JPM task brief
LGD = 1 - RECOVERY_RATE  # 0.90

def expected_loss(loan_properties: dict):
    """
    loan_properties: dict with keys: credit_line, loan_amt, total_debt, income, years_emp, fico_score
    EAD = loan_amt
    Returns: [Expected_Loss_$, PD]
    """
    ead = loan_properties['loan_amt']
    features_df = pd.DataFrame([loan_properties])
    
    if use_scaler:
        features_scaled = scaler.transform(features_df)
        pd_prob = best_model.predict_proba(features_scaled)[:, 1][0]
    else:
        pd_prob = best_model.predict_proba(features_df)[:, 1][0]
    
    el = pd_prob * ead * LGD
    return round(el, 2), round(pd_prob, 4)

# ========== 4. EXAMPLE: NEW BORROWER ==========
new_borrower = {
    'credit_line': 1,        # 0 or 1
    'loan_amt': 5000.00,     # EAD
    'total_debt': 8000.00, 
    'income': 65000.00,
    'years_emp': 4,
    'fico_score': 680
}

loss_dollars, pd_value = expected_loss(new_borrower)
print(f"\nNew Borrower PD: {pd_value}")
print(f"Expected Loss @10% Recovery: ${loss_dollars:,.2f}")
