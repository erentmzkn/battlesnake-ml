# features.py
# extrahiert Merkmale aus dem Spielfeld für das ML-Modell
# gibt eine Liste mit genau 6 Features zurück

def features_from_board(game_state):
    head = game_state["you"]["body"][0]
    health = game_state["you"]["health"]
    board = game_state["board"]

    head_x = head["x"]
    head_y = head["y"]
    width = board["width"]
    height = board["height"]

    food = board.get("food", [])
    food_dist = min(
        [abs(head_x - f["x"]) + abs(head_y - f["y"]) for f in food],
        default=width + height  # falls kein Futter: maximale Entfernung
    )

    return [head_x, head_y, health, width, height, food_dist]
