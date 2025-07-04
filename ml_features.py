# Gruppe 3 – Battlesnake Projekt (SS2025)
# Mitglieder:
# Eren Temizkan, 223201982
# Dominik Ide, 220200046
# Dogukan Karakoyun, 223202023
# Alexandra Holsten, 221200813
# Yuxiao Wu, 223200006


# eigentlich reine Kopie(aus dem Main-projekt)

def features_from_board(game_state):
    # Hole Kopfposition (unsere aktuelle Position)
    head = game_state["you"]["body"][0]

    # Lebenspunkte (starten bei 100, sinken jede Runde, Futter = Auffüllen)
    health = game_state["you"]["health"]

    # Spielfeldinfos
    board = game_state["board"]
    width = board["width"]
    height = board["height"]

    # Koordinaten vom Kopf
    head_x = head["x"]
    head_y = head["y"]

    # Futterpositionen (kann leer sein)
    food = board.get("food", [])

    # Manhattan-Distanz zum nächsten Futter berechnen
    food_dist = min(
        [abs(head_x - f["x"]) + abs(head_y - f["y"]) for f in food],
        default=width + height  # wenn kein Futter da → max. Distanz annehmen
    )

    # Debug-Ausgabe zum Checken, ob alles richtig extrahiert wurde
    print(f"[DEBUG] Features extrahiert: head=({head_x},{head_y}), health={health}, board=({width}x{height}), food_dist={food_dist}")

    # Finaler Feature-Vektor
    return [head_x, head_y, health, width, height, food_dist]
