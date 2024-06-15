import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_logs(player_types, rewards_log, actions_log, folder_path):
    epochs = rewards_log[list(rewards_log)[0]].shape[0]

    # Plot rewards log
    plt.figure(figsize=(12, 6))
    for key, value in player_types.items():
        plt.plot(range(epochs), rewards_log[key]/value, label=key)
    plt.title("Testing Rewards Log")
    plt.xlabel("Games")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.savefig(f"{folder_path}/testing_rewards.png")
    plt.close()

    # Plot actions log
    plt.figure(figsize=(12, 6))
    for key, value in player_types.items():
        plt.plot(range(epochs), actions_log[key]/value, label=key)
    plt.title("Testing Actions Log")
    plt.xlabel("Epoch")
    plt.ylabel("Average Action")
    plt.legend()
    plt.savefig(f"{folder_path}/testing_actions.png")
    plt.close()