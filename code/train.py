import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from game import create_game

def train(max_steps, plot):

    # Init Logs
    epochs = int(1e05)
    plot_frequency = 200
    # epochs = int(2e00)
    # max_steps = 5
    # plot_frequency = 1

    prob_return = False

    n_learners = 4
    fixed_players = {  # type_player:n_players
        # "Cu": 1,
        # "Du": 1,
        # "Random": 1,
        # "TFT": 1,
        # "STFT": 1,
        # "TF2T": 1,
    }
    n_history = 3

    agents, world = create_game(n_learners, fixed_players, n_history, max_steps, prob_return)

    player_types = fixed_players.copy()
    if n_learners > 1:
        for i in range(n_learners):
            player_types[f"RL{i+1}"] = 1
    elif n_learners > 0:
        player_types["RL"] = 1

    rewards_log = {key: np.zeros(epochs // plot_frequency) for key in player_types.keys()}

    if plot:
        plt.figure(figsize=(10, 6))

    # Train
    for agent in agents[:n_learners]:
        agent.brain.mode = "Training"

    for epoch in tqdm(range(epochs), desc="Training Progress"):

        # Play trajectory
        state = world.reset()
        while True:

            # choose actions
            actions = [agent.response(state[agent.opponent]) for i, agent in enumerate(agents)]

            # Step through world
            next_state, rewards, done = world.step(actions)

            # save transition to the replay memory
            for i, (agent, reward, action) in enumerate(zip(agents[:n_learners], rewards[:n_learners], actions[:n_learners])):
                agent.brain.memory.append(state[agent.opponent].copy(), action, reward, done, next_state[agent.opponent].copy())

            # Log
            for i, agent in enumerate(agents):
                rewards_log[agent.type][epoch // plot_frequency] += np.array(rewards[i]) / plot_frequency
            state = next_state.copy()

            if done:
                break

        # Train
        for agent in agents[:n_learners]:
            agent.train() 

        # Plot rewards log
        if plot and not (epoch+1) % plot_frequency:
            plt.clf()
            plt.xlabel("Games")
            plt.ylabel("Average Reward")
            # plt.ylim(bottom=0, top=max_steps*5+0.1)
            plt.ylim(bottom=max_steps*2-0.1, top=max_steps*6+0.1)
            plt.title("Training Progress")
            plt.grid()
            for key, value in player_types.items():
                plt.plot(range(1, epoch, plot_frequency), rewards_log[key][:epoch // plot_frequency + 1]/value, label=key)
            plt.legend()
            plt.draw()
            plt.pause(0.001)

    if n_learners > 0:
        mapping = {
            0: "D",
            1: "C",
            2: "S"
        }
        print("Learned Q")
        for index, element in np.ndenumerate(agents[0].brain.Q):
            index = "".join([mapping.get(x, x) for x in list(index)])
            print(f"Case {index}: {element}")

        print("Counts")
        for index, element in np.ndenumerate(agents[0].brain.count):
            index = "".join([mapping.get(x, x) for x in list(index)])
            print(f"Case {index}: {element}")

    # Inference
    for agent in agents[:n_learners]:
        agent.brain.mode = "Inference"

    if plot:
        path = "plots/"
        game_type = ",".join(list(rewards_log))
        folder_path = path + "Agents:" + game_type
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        plt.savefig(f"{folder_path}/training_rewards.png")
        plt.close()

    return n_learners, player_types, agents, world, folder_path