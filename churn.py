import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from datetime import datetime

# ✅ Load all CSVs
customer_data = pd.read_csv('customer_data.csv')
transactions = pd.read_csv('transactions.csv')
support_tickets = pd.read_csv('support_tickets.csv')
web_activity = pd.read_csv('web_activity.csv')

# ✅ Transactions metrics — fix column names!
transaction_stats = transactions.groupby('CustomerID').agg({
    'Amount': ['mean', 'sum'],
    'Date': ['count', 'max']  # ✅ Correct column name!
})
transaction_stats.columns = ['AvgTransactionAmount', 'TotalSpend', 'TransactionCount', 'LastTransactionDate']

# ✅ Days since last transaction
transaction_stats['DaysSinceLastTransaction'] = (
    datetime.now() - pd.to_datetime(transaction_stats['LastTransactionDate'])
).dt.days

# ✅ Support tickets
ticket_stats = support_tickets.groupby('CustomerID').agg({
    'TicketID': 'count'
}).rename(columns={'TicketID': 'TicketCount'})

# ✅ Web activity — fix column name!
web_stats = web_activity.groupby('CustomerID').agg({
    'PagesViewed': 'mean',
    'SessionDurationMinutes': 'mean'  # ✅ Correct column name!
}).rename(columns={'PagesViewed': 'AvgPagesViewed', 'SessionDurationMinutes': 'AvgSessionDuration'})

# ✅ Join all
enriched_data = customer_data.merge(transaction_stats, on='CustomerID', how='left')
enriched_data = enriched_data.merge(ticket_stats, on='CustomerID', how='left')
enriched_data = enriched_data.merge(web_stats, on='CustomerID', how='left')

# ✅ Flags
enriched_data['HighTicketCount'] = (enriched_data['TicketCount'] > 3).astype(int)
enriched_data['LowActivity'] = (enriched_data['AvgPagesViewed'] < 3).astype(int)
enriched_data['InactiveCustomer'] = (enriched_data['DaysSinceLastTransaction'] > 30).astype(int)

# ✅ Fill missing
enriched_data.fillna({
    'AvgTransactionAmount': 0,
    'TotalSpend': 0,
    'TransactionCount': 0,
    'DaysSinceLastTransaction': 365,
    'TicketCount': 0,
    'AvgPagesViewed': 0,
    'AvgSessionDuration': 0,
    'HighTicketCount': 0,
    'LowActivity': 0,
    'InactiveCustomer': 1
}, inplace=True)

# ✅ Features & Target — fix target name!
features = enriched_data[[ 
    'AvgTransactionAmount', 'TotalSpend', 'TransactionCount', 'DaysSinceLastTransaction',
    'TicketCount', 'AvgPagesViewed', 'AvgSessionDuration',
    'HighTicketCount', 'LowActivity', 'InactiveCustomer'
]]
target = enriched_data['Churn']  # ✅ Correct column name!

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred))

# ✅ Save
joblib.dump(model, 'customer_churn_model.joblib')

# ✅ Scoring
def score_new_data(customer_ids):
    model = joblib.load('customer_churn_model.joblib')

    # ✅ Filter for new IDs
    customers = customer_data[customer_data['CustomerID'].isin(customer_ids)].copy()

    if customers.empty:
        raise ValueError(f"No matching customers found for {customer_ids}")

    # ✅ Recompute placeholders — you must replace these with real new data if available!
    customers['AvgTransactionAmount'] = 0
    customers['TotalSpend'] = 0
    customers['TransactionCount'] = 0
    customers['DaysSinceLastTransaction'] = 365
    customers['TicketCount'] = 0
    customers['AvgPagesViewed'] = 0
    customers['AvgSessionDuration'] = 0
    customers['HighTicketCount'] = (customers['TicketCount'] > 3).astype(int)
    customers['LowActivity'] = (customers['AvgPagesViewed'] < 3).astype(int)
    customers['InactiveCustomer'] = (customers['DaysSinceLastTransaction'] > 30).astype(int)

    features_new = customers[[ 
        'AvgTransactionAmount', 'TotalSpend', 'TransactionCount', 'DaysSinceLastTransaction',
        'TicketCount', 'AvgPagesViewed', 'AvgSessionDuration',
        'HighTicketCount', 'LowActivity', 'InactiveCustomer'
    ]]

    predictions = model.predict_proba(features_new)[:, 1]

    return pd.DataFrame({
        'CustomerID': customers['CustomerID'],
        'ChurnProbability': predictions,
        'ChurnRisk': np.where(predictions > 0.5, 'High', 'Low')
    })

# ✅ Example — use real IDs!
print("Available IDs:", customer_data['CustomerID'].tolist())
new_customers = [customer_data['CustomerID'].iloc[0]]  # Use an actual ID
result = score_new_data(new_customers)
print("\nChurn Predictions:\n", result)
