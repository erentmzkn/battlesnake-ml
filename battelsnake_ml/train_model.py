
import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import matplotlib.pyplot as plt

# Base vom Root
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "training_data.csv"
model_path = base_dir / "model.pkl"

# Prüfen, ob Trainingsdatei existiert
if not csv_path.exists():
    print("[INFO] 'training_data.csv' nicht gefunden – leere Datei wird erstellt.")
    df = pd.DataFrame(columns=[
        "head_x", "head_y", "health", "width", "height", "closest_food_distance", "move"
    ])
    df.to_csv(csv_path, index=False)
    print("[INFO] Leere Datei erstellt. Bitte zuerst Trainingsdaten sammeln.")
    exit()

# Trainingsdaten laden
print("[DEBUG] Lade Trainingsdaten...")
data = pd.read_csv(csv_path)
print(f"[DEBUG] Zeilen geladen: {len(data)}\n")

if data.empty:
    print("[WARNUNG] Datei ist leer –> Spiele erstmal ein paar Spiele.")
    exit()

# Features und Ziel trennen
X = data.drop("move", axis=1)
y = data["move"]

# Verteilung anzeigen
print("Züge-Verteilung:")
print(y.value_counts())
print()

# Trainings-/Testdaten splitten
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[DEBUG] Training: {len(X_train)} | Test: {len(X_test)}\n")

# Modell erstellen & trainieren
print("Training läuft...")
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)
print("Training abgeschlossen!\n")

#  Klassifikationsübersicht
print("Klassifikationsreport:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature-Wichtigkeit anzeigen
print("[DEBUG] Zeige Featurewichtigkeit...")
importances = clf.feature_importances_
feature_names = X.columns.tolist()

plt.barh(feature_names, importances)
plt.xlabel("Wichtigkeit")
plt.title("Feature-Wichtigkeit des Modells")
plt.tight_layout()
plt.savefig("feature_importance.png")  # Wird als Bild gespeichert
print("[INFO] Feature-Wichtigkeit gespeichert als 'feature_importance.png'.")

# Modell speichern
dump(clf, model_path)
print(f"Modell gespeichert als: '{model_path}'")


