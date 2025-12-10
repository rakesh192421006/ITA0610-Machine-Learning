# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load dataset
dataset = pd.read_csv("HousePricePrediction.csv")

# 3. Basic exploration
print("Shape:", dataset.shape)
print(dataset.head())
print(dataset.info())

# 4. Drop unnecessary columns (like Id)
if 'Id' in dataset.columns:
    dataset.drop(['Id'], axis=1, inplace=True)

# 5. Handle missing values separately for numeric and categorical
numeric_cols = dataset.select_dtypes(include=[np.number]).columns
categorical_cols = dataset.select_dtypes(include=['object']).columns

# Fill numeric NaN with column mean
dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())

# Fill categorical NaN with column mode
for col in categorical_cols:
    dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

# 6. Encode categorical features using one-hot encoding
dataset = pd.get_dummies(dataset, drop_first=True)

# 7. Split features and target
X = dataset.drop("SalePrice", axis=1)
y = dataset["SalePrice"]

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 10. Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 11. Evaluation function
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Performance:")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")
    print("-"*40)

# 12. Evaluate both models
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# 13. Feature importance (Random Forest)
importance = rf_model.feature_importances_
feat_importance = pd.DataFrame({"Feature": X.columns, "Importance": importance})
feat_importance = feat_importance.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_importance.head(10))
plt.title("Top 10 Features Influencing House Price")
plt.show()
