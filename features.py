# Gruppe 3 – Battlesnake Projekt (SS2025)
# Mitglieder:
# Eren Temizkan, 223201982
# Dominik Ide, 220200046
# Dogukan Karakoyun, 223202023
# Alexandra Holsten, 221200813
# Yuxiao Wu, 223200006


# reine Kopie
# zieht ein paar Basismerkmale aus dem Spielfeld raus
# später könnte eventual erweitern mit mehr Kontext, z.B. Gegnerdistanz, Mauern, etc.

def features_from_board(game_state):
    # Hole Kopfposition (unsere aktuelle Position auf dem Spielfeld)
    head = game_state["you"]["body"][0]

    # Lebenspunkte (starten bei 100, sinken jede Runde, +Futter = Auffüllen)
    health = game_state["you"]["health"]

    # Das gesamte Spielfeldobjekt
    board = game_state["board"]

    # Extrahiere Position und Spielfeldgröße
    head_x = head["x"]
    head_y = head["y"]
    width = board["width"]
    height = board["height"]

    # Alle Futterpositionen (kann ja auch leer sein)
    food = board.get("food", [])

    # Berechne Manhattan-Distanz zum nächsten Futter (formel)
    food_dist = min(
        [abs(head_x - f["x"]) + abs(head_y - f["y"]) for f in food],
        default=width + height  # wenn kein Futter da ist → max. Distanz annehmen
    )

    # Feature Vektor bestehend aus 6 fixen Werten → wichtig für Modell Input
    return [head_x, head_y, health, width, height, food_dist]
