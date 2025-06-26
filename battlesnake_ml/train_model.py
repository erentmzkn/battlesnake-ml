import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("training_data.csv")
X = data.drop("move", axis=1)
y = data["move"]

# Show class distribution
print("Move distribution:\n", y.value_counts())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance plot
importances = clf.feature_importances_
feature_names = X.columns.tolist()
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Save model
dump(clf, "model.pkl")
print("✅ Model trained and saved as model.pkl")
