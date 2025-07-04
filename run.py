from flask import Flask, request, jsonify
import pandas as pd
import traceback
from joblib import load
from pathlib import Path
from ml_features import features_from_board

app = Flask(__name__)

# Modell laden
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "model.pkl"

# Modell laden
try:
    model = load(model_path)
    print(f"[ML] model.pkl erfolgreich geladen von: {model_path}")
except Exception as e:
    print(f"[ERROR] Modell konnte nicht geladen werden: {e}")
    model = None

@app.route("/start", methods=["POST"])
def start():
    print("[INFO] Game started.")
    return "ok"

@app.route("/end", methods=["POST"])
def end():
    print("[INFO] Game ended.")
    return "ok"

@app.route("/move", methods=["POST"])
def move():
    try:
        game_state = request.get_json()
        features = features_from_board(game_state)

        print(f"[DEBUG] Eingabefeatures: {features}")

        if model:
            try:
                # Richtige Spaltennamen verwenden (wichtig!)
                columns = ["head_x", "head_y", "health", "width", "height", "closest_food_distance"]
                features_df = pd.DataFrame([features], columns=columns)
                prediction = model.predict(features_df)[0]
                print(f"[ML] Vorhergesagter Zug: {prediction}")
                return jsonify({"move": prediction})
            except Exception as ml_error:
                print(f"[WARNUNG] ML-Vorhersage fehlgeschlagen: {ml_error}")

    except Exception as e:
        print("[ERROR] Fehler im Move-Handler:")
        traceback.print_exc()

    # Fallback-Zug
    fallback_move = "up"
    print(f"[Fallback] Fallback-Zug gewählt: {fallback_move}")
    return jsonify({"move": fallback_move})


if __name__ == "__main__":
    print("[INFO] Starte Battlesnake-Server...")
    app.run(host="0.0.0.0", port=8000, debug=True)


