# features.py
from features import features_from_board

def features_from_board(game_state):
    head = game_state["you"]["body"][0]
    health = game_state["you"]["health"]
    board = game_state["board"]

    head_x = head["x"]
    head_y = head["y"]

    food = board["food"]
    food_dist = min(
        [abs(head_x - f["x"]) + abs(head_y - f["y"]) for f in food],
        default=0
    )

    return [head_x, head_y, health, food_dist]

