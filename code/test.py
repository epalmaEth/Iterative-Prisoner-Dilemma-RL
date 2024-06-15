import numpy as np
from tqdm import tqdm

def test(agents, world, player_types, max_steps):

    # Init Logs
    epochs = int(1e02)
    
    rewards_log = {key: np.zeros(epochs) for key in player_types.keys()}
    actions_log = {key: np.zeros(epochs) for key in player_types.keys()}

    for epoch in tqdm(range(epochs), desc="Inference Progress"):

        # Play trajectory
        state = world.reset()
        while True:
            
            # choose actions
            actions = [agent.response(state[agent.opponent]) for i, agent in enumerate(agents)]

            # Step through world
            next_state, rewards, done = world.step(actions)
            # Log
            for i, agent in enumerate(agents):
                rewards_log[agent.type][epoch] += np.array(rewards[i])
                actions_log[agent.type][epoch] += np.array(actions[i]) / max_steps
            state = next_state

            if done:
                break

    return rewards_log, actions_log