# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 2. Load dataset (Iris dataset)
iris = pd.read_csv("IRIS.csv")
print(iris.head())

# 3. Features and target
X = iris.drop("species", axis=1)
y = iris["species"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize classifiers
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42)
}

# 6. Train & evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Algorithm": name, "Accuracy": acc})
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred))

# 7. Compare results in a DataFrame
results_df = pd.DataFrame(results)
print("\nComparison of Classification Algorithms:\n")
print(results_df)

# 8. Visualization
plt.figure(figsize=(10,6))
sns.barplot(x="Accuracy", y="Algorithm", data=results_df, palette="viridis")
plt.title("Comparison of Classification Algorithms on Iris Dataset")
plt.show()
