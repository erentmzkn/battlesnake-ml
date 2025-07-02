import pandas as pd  # Zum Einlesen und Verarbeiten von CSV-Daten
from sklearn.ensemble import RandomForestClassifier  # Klassifikator auf Basis vieler Entscheidungsbäume
from sklearn.model_selection import train_test_split  # Zum Aufteilen der Daten in Trainings- und Testmenge
from sklearn.metrics import classification_report  # Für die Auswertung des Modells
from joblib import dump  # Ermöglicht das Speichern des trainierten Modells
import matplotlib.pyplot as plt  # Für die Visualisierung der Feature-Wichtigkeit
from pathlib import Path  # Um Dateipfade robust zu definieren

# Basisverzeichnis bestimmen (also battlesnake-ml Ordner)
base_dir = Path(__file__).resolve().parent.parent

# Schritt 1: Daten aus CSV-Datei laden
# erwartet wird eine Datei mit Features und einer Zielspalte ("move")
csv_path = base_dir / "training_data.csv"
data = pd.read_csv(csv_path)

# Schritt 2: Eingabedaten (X) und Zielwerte (y) definieren
print("Spalten im DataFrame:", data.columns.tolist()) #debugging purposes
X = data.drop("move", axis=1)  # Alle Spalten außer 'move' sind Eingabemerkmale
# entferne sie die Spalte namens "move" aus "data", weil move(gespeichert als X) das ist, was wir vorhersagen können
# wollen. Damit wird X zu einer 2D-Eingabematrix[X, 6]. Also wollen wir nicht, dass unser Modell "move" als eine
# Eingabe sieht, sondern als eine Zielvariable, damit es nicht die Antwort merkt, sondern lernt.
y = data["move"]  # Zielvariable ist die Zugrichtung
# #Also Y="was wir lernen wollen" (Y ist auch ein 1D Array), damit X die Inputs und Features sind, wobei Y outputs
# und labels.

# Schritt 3: Ausgabe der Klassenverteilung.
# So erkennt man, ob bestimmte Züge überrepräsentiert oder unterrepräsentiert sind
print("Verteilung der Züge:\n", y.value_counts())

# Schritt 4: Aufteilen in Trainings- und Testdaten
# 80 % werden für das Training verwendet, 20 % für die spätere Evaluation
X_train, X_test, y_train, y_test = train_test_split( # Training-set und Test-set, wobei X_train = Eingabe-Matrix,
    # Y_train = Deren entsprechenden korrekten Moves, X_test = Test-Matrix, Y_test = Die korrekten Moves in dem Testset
    X, y, test_size=0.2, random_state=77 # 0.2 = 20% der Daten für Test und damit 80 % für Training, random_state = 77
    # für Reproduzierbarkeit implementiert durch "from sklearn.model_selection import train_test_split"
) # wobei X_train = Eingabemerkmale, X_test = Testmerkmale, y_train = Zielvariablen, y_test = Testzielvariablen

# Schritt 5: Initialisierung und Training des Random Forest-Klassifikators
# class_weight="balanced" gleicht ungleiche Klassenverteilungen aus
clf = RandomForestClassifier( # Classifier-Modell
    n_estimators=100,  # Anzahl der Entscheidungsbäume
    class_weight="balanced",  # Gewichtung der Klassen je nach Häufigkeit, "balanced", damit die Klassen nicht biased
    # werden wenn einige Moves weniger auftreten.
    random_state=77  # Für Reproduzierbarkeit, damit immer die gleichen Reihen fürs Training oder Testing gewählt
    # werden.
)
clf.fit(X_train, y_train)  # Das Modell wird auf den Trainingsdaten trainiert, wobei jeder Baum im Wald 80 % der
# Reihen in X_train mit deren entsprechenden richtigen Moves im y_train sieht, zufallige Reihen und Features
# auswählt, danach eine "if this, then that" Beziehung bildet, dann versucht den richtigen Move für jede Zelle zu
# bestimmen.

# Schritt 6: Vorhersage und Auswertung mit den Testdaten
y_pred = clf.predict(X_test)  # Vorhersage auf Basis der Testdaten, y_pred = "Moves das Modell denkt ist richtig"
# versus y_test = "Moves die eigentlich richtig sind aus der CSV-Datei"
# Ausgabe von Metriken wie Präzision, Recall, F1-Score
print("\nKlassifikationsbericht:\n", classification_report(y_test, y_pred)) # Vergleich der beiden Arrays

# Schritt 7: Visualisierung der Wichtigkeit einzelner Features
# ermittelt, welche Eingabe-Variablen am meisten zur Entscheidung beitragen
importances = clf.feature_importances_  # Array mit Wichtigkeiten. D.h, es zeigt, welche X-Matrix-Eingabemerkmale am
# meisten zur Entscheidung beigetragen haben
feature_names = X.columns.tolist()  # Namen der Eingabemerkmale

# Balkendiagramm der Wichtigkeiten
plt.barh(feature_names, importances) # Zeige es auch als Balkendiagramm an
plt.xlabel("Wichtigkeit") # Balkendiagramm: x-Achse: Wichtigkeit
plt.title("Feature-Wichtigkeit des Modells") # Balkendiagramm: y-Achse: Namen der Eingabemerkmale
plt.tight_layout()  # Vermeidet abgeschnittene Achsenbeschriftungen
plt.show()

# Schritt 8: Speichern des trainierten Modells als .pkl-Datei
model_path = base_dir / "model.pkl"
dump(clf, model_path)  # Serialisierung mit joblib(um unser Modell zu speichern und zu laden), damit unser
# training-Modell(RandomForestClassifier Entität) als .pkl-Datei gespeichert wird
print(f"Modell wurde trainiert und unter '{model_path}' gespeichert.")

