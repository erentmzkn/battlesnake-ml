from features import features_from_board  # Import der Feature-Extraktionsfunktion (bei Modulstruktur erforderlich)

# Funktion zur Extraktion relevanter Spielfeldinformationen aus dem game_state-Dictionary
def features_from_board(game_state):
    # Extrahieren der Position des Schlangenkopfs (erste Koordinate im Körperarray)
    head = game_state["you"]["body"][0]  # 'you' bezieht sich auf die eigene Schlange
    head_x = head["x"]  # X-Koordinate des Kopfs
    head_y = head["y"]  # Y-Koordinate des Kopfs

    # Lebenspunkte der Schlange (zwischen 0 und 100)
    health = game_state["you"]["health"]

    # Spielfeldinformationen (inkl. Größe, andere Schlangen, Futterpositionen)
    board = game_state["board"]

    # Liste der Futterpositionen auf dem Spielfeld
    food = board["food"]

    # Berechnung der Manhattan-Distanz zum nächsten Futterstück.
    # Falls kein Futter vorhanden ist, wird standardmäßig der Abstand 0 verwendet
    food_dist = min( #Formel zur Manhattan-Distanz zwischen Schlangenkopf und nächstem Futter
        [abs(head_x - f["x"]) + abs(head_y - f["y"]) for f in food],
        default=0  # falls food leer ist
    )

    # Rückgabe der extrahierten Features in einer Liste.
    # Diese Werte werden später als Eingabe für das ML-Modell verwendet
    return [head_x, head_y, health, food_dist]


