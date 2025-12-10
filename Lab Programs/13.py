# Car Price Prediction Model using Python

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 2. Load dataset
data = pd.read_csv("CarPrice.csv")

# 3. Basic exploration
print("Shape:", data.shape)
print("Columns:", data.columns)
print(data.info())
print(data.describe())

# 4. Handle categorical features
# Convert categorical columns into numeric using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# 5. Correlation analysis (numeric only)
plt.figure(figsize=(20, 15))
sns.heatmap(data_encoded.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Car Price Dataset")
plt.show()

# 6. Feature selection
X = data_encoded.drop("price", axis=1)   # Features
y = data_encoded["price"]                # Target

# 7. Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(xtrain, ytrain)

# 9. Predictions
predictions = model.predict(xtest)

# 10. Evaluation
mae = mean_absolute_error(ytest, predictions)
r2 = r2_score(ytest, predictions)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# 11. Feature importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Optional: Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Features Influencing Car Price")
plt.show()
