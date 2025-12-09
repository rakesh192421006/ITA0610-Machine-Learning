import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("CREDITSCORE.csv")
print(data.head())
print(data.info())

# Encode Credit_Mix to numeric values
data["Credit_Mix"] = data["Credit_Mix"].map({
    "Bad": 0,
    "Standard": 1,
    "Good": 3
})

# If there are any missing values in Credit_Mix after mapping, fill with 1 (Standard)
data["Credit_Mix"] = data["Credit_Mix"].fillna(1)

# Feature matrix (X) and target vector (y)
X = data[[
    "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Credit_Mix", "Outstanding_Debt",
    "Credit_History_Age", "Monthly_Balance"
]].values

y = data["Credit_Score"].values

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Model training
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# Prediction for a new user
print("\n--- Credit Score Prediction ---")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = float(input("Credit Mix (Bad: 0, Standard: 1, Good: 3): "))
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
prediction = model.predict(features)

print("\nPredicted Credit Score =", prediction[0])
