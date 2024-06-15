def key_encoder(key):
    if key == "RL":
        return 0
    return 1

def select_player_type(players_types, i_player):
    for key, value in players_types.items():
        i_player -= value
        if i_player < 0:
            return key_encoder(key)
    return None