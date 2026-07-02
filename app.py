import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

st.set_page_config(page_title="JPM Credit Risk Task 3", layout="wide")

RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE # 0.90

@st.cache_data
def train_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "Task 3 and 4_Loan_Data.csv")
    
    df = pd.read_csv(file_path)
    feature_cols = ['credit_line', 'loan_amt', 'total_debt', 'income', 'years_emp', 'fico_score'] # FIX 1: Lock order
    X = df[feature_cols] 
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    
    return model, auc, feature_cols # FIX 2: Return the columns

model, auc, feature_cols = train_model()

st.title("JPMorgan Forage: Task 3 Credit Risk")
st.metric("Model AUC on Test Set", f"{auc:.4f}")

st.subheader("Enter New Borrower Details to get PD + Expected Loss")

col1, col2 = st.columns(2)
with col1:
    credit_line = st.selectbox("Has Credit Line?", [0, 1])
    loan_amt = st.number_input("Loan Amount EAD $", min_value=100.0, value=5000.00, step=100.0, format="%.2f")
    total_debt = st.number_input("Total Debt $", min_value=0.0, value=8000.00, step=100.0, format="%.2f")
    income = st.number_input("Annual Income $", min_value=0.0, value=65000.00, step=1000.0, format="%.2f")
with col2:
    years_emp = st.number_input("Years Employed", min_value=0.0, value=4.0, step=1.0, format="%.0f")
    fico_score = st.number_input("FICO Score", min_value=300.0, max_value=850.0, value=680.0, step=1.0, format="%.0f")

if st.button("Calculate Expected Loss"):
    # FIX 3: Build list in the EXACT order sklearn expects
    input_values = [
        float(credit_line),
        float(loan_amt), 
        float(total_debt),
        float(income),
        float(years_emp),
        float(fico_score)
    ]
    
    input_df = pd.DataFrame([input_values], columns=feature_cols) # <-- Use saved columns
    
    pd_prob = model.predict_proba(input_df)[:, 1][0]
    expected_loss = pd_prob * loan_amt * LGD
    
    st.success(f"**Predicted PD: {pd_prob:.4f}**")
    st.metric(label=f"Expected Loss @10% Recovery", value=f"${expected_loss:,.2f}")
    st.caption('Formula used: EL = PD * EAD * LGD, where LGD = 1 - 0.10 = 0.90')
