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

# Drop ID, use 'import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

st.set_page_config(page_title="JPM Credit Risk Task 3", layout="wide")

RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE # 0.90

@st.cache_data
def train_model():
    """Train PD model on the uploaded CSV"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "Task 3 and 4_Loan_Data.csv")
    
    df = pd.read_csv(file_path)
    X = df.drop(columns=['customer_id', 'default']) 
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train) # RF doesn't need scaling, but we'll use it for consistency
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    
    return model, scaler, auc, X.columns.tolist()

model, scaler, auc, feature_cols = train_model()

# ========== UI ==========
st.title("JPMorgan Forage: Task 3 Credit Risk")
st.metric("Model AUC on Test Set", f"{auc:.4f}")

st.subheader("Enter New Borrower Details to get PD + Expected Loss")

col1, col2, col3 = st.columns(3)
with col1:
    credit_line = st.selectbox("Has Credit Line?", [0, 1])
    loan_amt = st.number_input("Loan Amount EAD $", min_value=100, value=5000.00, step=100.0)
    total_debt = st.number_input("Total Debt $", min_value=0.0, value=8000.00, step=100.0)
with col2:
    income = st.number_input("Annual Income $", min_value=0.0, value=65000.00, step=1000.0)
    years_emp = st.number_input("Years Employed", min_value=0, value=4, step=1)
    fico_score = st.number_input("FICO Score", min_value=300, max_value=850, value=680, step=1)

if st.button("Calculate Expected Loss"):
    input_dict = {
        'credit_line': credit_line,
        'loan_amt': loan_amt,
        'total_debt': total_debt,
        'income': income,
        'years_emp': years_emp,
        'fico_score': fico_score
    }
    
    input_df = pd.DataFrame([input_dict])
    # RF trained without scaler, so predict directly
    pd_prob = model.predict_proba(input_df)[:, 1][0]
    expected_loss = pd_prob * loan_amt * LGD
    
    st.success(f"**Predicted PD: {pd_prob:.4f}**")
    st.metric(label=f"Expected Loss @10% Recovery", value=f"${expected_loss:,.2f}")
    
    st.caption("Formula used: EL = PD * EAD * LGD, where LGD = 1 - 0.10 = 0.90")default' as target
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
