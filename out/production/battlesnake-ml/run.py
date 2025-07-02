import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Klassifikator auf Basis vieler Entscheidungsbäume
from sklearn.model_selection import train_test_split  # Für das Aufteilen der Daten in Trainings- und Testdaten
from sklearn.metrics import classification_report  # Für die Evaluierung des Modells
from joblib import dump  # Zum Speichern des trainierten Modells

# Pfad zur CSV-Datei mit den gesammelten Trainingsdaten
csv_path = "training_data.csv"

# Schritt 1: Prüfen, ob die CSV-Datei existiert. Wenn nicht, wird sie mit passenden Spaltenüberschriften angelegt.
if not os.path.exists(csv_path):
    print("Keine training_data.csv gefunden. Erstelle eine leere Datei mit Spaltenüberschriften.")
    df = pd.DataFrame(columns=["head_x", "head_y", "health", "width", "height", "closest_food_distance", "move"])
    df.to_csv(csv_path, index=False)

# Schritt 2: CSV-Datei laden
data = pd.read_csv(csv_path)

# Schritt 3: Prüfen, ob Daten vorhanden sind. Falls leer, abbrechen.
if data.empty:
    print("Warnung: training_data.csv ist leer. Spiele zuerst ein paar Runden, um Trainingsdaten zu erzeugen.")
    exit()

# Schritt 4: Aufteilen in Eingabemerkmale (X) und Zielvariable (y)
# Die Spalte "move" ist die Zielvariable, die vorhergesagt werden soll.
X = data.drop("move", axis=1)  # Alle Spalten außer "move" sind Merkmale
y = data["move"]  # Zielvariable: Richtung, in die sich die Schlange bewegt hat

# Schritt 5: Ausgabe der Klassenverteilung, um zu sehen, wie oft jede Bewegung vorkommt
print("\nVerteilung der Züge:")
print(y.value_counts())

# Schritt 6: Aufteilen der Daten in Trainings- und Testdaten (80 % Training, 20 % Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schritt 7: Modell initialisieren und auf den Trainingsdaten trainieren
# class_weight="balanced" hilft dabei, ungleiche Klassenverteilungen auszugleichen
print("\nTrainiere das Modell...")
model = RandomForestClassifier(class_weight="balanced")
model.fit(X_train, y_train)

# Schritt 8: Evaluierung des Modells mit den Testdaten
# Zeigt Metriken wie Präzision, Recall und F1-Score
print("\nEvaluierung des Modells:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Schritt 9: Das trainierte Modell als .pkl-Datei speichern
model_path = "model.pkl"
dump(model, model_path)
print(f"\nModell wurde erfolgreich unter '{model_path}' gespeichert.")

