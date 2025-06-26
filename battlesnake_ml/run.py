import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

csv_path = "training_data.csv"

# ✅ Step 1: Check if file exists
if not os.path.exists(csv_path):
    print("⚠️ No training_data.csv found. Creating a blank one with headers.")
    df = pd.DataFrame(columns=["head_x", "head_y", "health", "width", "height", "closest_food_distance", "move"])
    df.to_csv(csv_path, index=False)

# 📥 Step 2: Load data
data = pd.read_csv(csv_path)

# 🛑 Step 3: If there's no data yet, stop
if data.empty:
    print("⚠️ training_data.csv is empty. Play some games first!")
    exit()

# 🧪 Step 4: Prepare features (X) and labels (y)
X = data.drop("move", axis=1)
y = data["move"]

# 🎯 Step 5: See how many examples of each move we have
print("\n🎯 Move distribution:")
print(y.value_counts())

# ✂️ Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Step 7: Train model
print("\n🚀 Training model...")
model = RandomForestClassifier(class_weight="balanced")
model.fit(X_train, y_train)

# 📊 Step 8: Evaluate
print("\n📊 Model evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 💾 Step 9: Save model
model_path = "model.pkl"
dump(model, model_path)
print(f"\n✅ Model saved as '{model_path}'")
