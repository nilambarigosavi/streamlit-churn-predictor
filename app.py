import streamlit as st
import joblib
import pandas as pd

# ------------------ Page Setup ------------------ #
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

# ------------------ Load Model ------------------ #
model = joblib.load('models/customer_churn_model.joblib')

def predict_churn(input_data):
    prediction = model.predict_proba(input_data)[:, 1]
    return prediction[0]

# ------------------ Custom CSS ------------------ #
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        color: #145A32; /* Change text color */
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        color: #145A32; /* Change text color in main container */
    }
    div.stButton > button {
        background-color: #007BFF;
        color: black;
        border: none;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #007BFF;
    }
    p, label, .stTextInput > label, .stSelectbox > label, .stNumberInput > label, .stMarkdown {
        color: #145A32 !important; /* Change color for most visible text */
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------ Sidebar ------------------ #
with st.sidebar:
    st.header("📌 About")
    st.write("""
    This tool predicts **customer churn** probability  
    based on your input data.
    """)
    st.write("🔗 Created by Nilambari Gosavi")
    st.write("📅 2025")

# ------------------ Title & Description ------------------ #
st.title("📊 Customer Churn Prediction")
st.write("Fill in the customer details below and click **Predict** to see churn probability.")

# ------------------ Input Form ------------------ #
with st.form("customer_form"):
    st.header("📝 Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        avg_transaction_amount = st.number_input("Avg Transaction Amount", min_value=0.0)
        total_spend = st.number_input("Total Spend", min_value=0.0)
        transaction_count = st.number_input("Transaction Count", min_value=0)
        days_since_last_transaction = st.number_input("Days Since Last Transaction", min_value=0)

    with col2:
        ticket_count = st.number_input("Ticket Count", min_value=0)
        avg_pages_viewed = st.number_input("Avg Pages Viewed", min_value=0.0)
        avg_session_duration = st.number_input("Avg Session Duration", min_value=0.0)
        high_ticket_count = st.selectbox("High Ticket Count (>3)?", [0, 1])
        low_activity = st.selectbox("Low Activity (<3 pages)?", [0, 1])
        inactive_customer = st.selectbox("Inactive Customer (>30 days)?", [0, 1])

    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_data = pd.DataFrame([[
            avg_transaction_amount, total_spend, transaction_count, days_since_last_transaction,
            ticket_count, avg_pages_viewed, avg_session_duration,
            high_ticket_count, low_activity, inactive_customer
        ]], columns=[
            'AvgTransactionAmount', 'TotalSpend', 'TransactionCount', 'DaysSinceLastTransaction',
            'TicketCount', 'AvgPagesViewed', 'AvgSessionDuration',
            'HighTicketCount', 'LowActivity', 'InactiveCustomer'
        ])

        proba = predict_churn(input_data)

        st.subheader("📌 Prediction Result")

        if proba > 0.5:
            st.error(f"⚠️ High churn risk! Probability: {proba:.1%}")
        else:
            st.success(f"✅ Low churn risk! Probability: {proba:.1%}")

# ------------------ Footer ------------------ #
st.markdown("---")

