import os  # Modul für Betriebssystem-Funktionen (z. B. Dateiüberprüfung)
import pandas as pd  # wird für die Datenverarbeitung verwendet
from sklearn.ensemble import RandomForestClassifier  # Klassifikator-Modell
from sklearn.model_selection import train_test_split  # Funktion zur Aufteilung in Trainings- und Testdaten
from sklearn.metrics import classification_report  # Für die Evaluierung des Modells
from joblib import dump  # Zum Speichern des trainierten Modells

# Pfad zur CSV-Datei mit den Trainingsdaten
csv_path = "training_data.csv"

# Schritt 1: Prüfen, ob die CSV-Datei bereits existiert
if not os.path.exists(csv_path):
    print("Warnung: training_data.csv nicht gefunden. Es wird eine leere Datei mit Spaltenüberschriften erstellt.")
    # Falls die Datei nicht existiert, erzeugen wir eine neue mit den notwendigen Spalten
    df = pd.DataFrame(columns=[
        "head_x",  # X-Position des Schlangenkopfs
        "head_y",  # Y-Position des Schlangenkopfs
        "health",  # Lebenspunkte der Schlange (0–100)
        "width",   # Breite des Spielfelds
        "height",  # Höhe des Spielfelds
        "closest_food_distance",  # Abstand zum nächsten Futter (euklidisch oder manhattan)
        "move"  # Zielwert: die Richtung (z. B. 'up', 'down', etc.)
    ])
    df.to_csv(csv_path, index=False)  # CSV-Datei ohne Index speichern

# Schritt 2: Laden der Daten aus der CSV-Datei
data = pd.read_csv(csv_path)

# Schritt 3: Überprüfung, ob Daten vorhanden sind
if data.empty:
    print("Die Datei training_data.csv ist leer. Bitte zuerst ein paar Spiele spielen, damit Trainingsdaten gesammelt werden.")
    exit()  # Programmabbruch, da Training ohne Daten keinen Sinn ergibt

# Schritt 4: Aufteilen in Eingabedaten (Features) und Zielwerte (Labels)
# X enthält alle Spalten außer 'move' (unsere erklärenden Variablen)
X = data.drop("move", axis=1)

# y enthält nur die 'move'-Spalte (unsere Zielvariable für die Klassifikation)
y = data["move"]

# Schritt 5: Ausgeben der Verteilung der Zielwerte
print("\nVerteilung der Klassen (Zugrichtungen):")
print(y.value_counts())  # Gibt aus, wie oft jede Bewegung im Datensatz vorkommt

# Schritt 6: Aufteilen in Trainings- und Testdaten
# 80 % der Daten für Training, 20 % für Evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Schritt 7: Erstellen und Trainieren des Klassifikators
print("\nModell wird trainiert...")
# RandomForestClassifier ist ein Ensemble-Modell basierend auf Entscheidungsbäumen
# class_weight="balanced" sorgt dafür, dass Klassen mit weniger Beispielen stärker gewichtet werden
model = RandomForestClassifier(class_weight="balanced")
model.fit(X_train, y_train)  # Training mit den Trainingsdaten

# Schritt 8: Auswertung des Modells mit den Testdaten
print("\nAuswertung des Modells:")
y_pred = model.predict(X_test)  # Vorhersagen für die Testdaten
# classification_report zeigt Metriken wie Präzision, Recall und F1-Score
print(classification_report(y_test, y_pred))

# Schritt 9: Speichern des trainierten Modells
model_path = "model.pkl"  # Zielpfad für das Modell
dump(model, model_path)  # Modell als .pkl-Datei abspeichern (Pickle-Format)
print(f"\nModell wurde gespeichert unter '{model_path}'")

