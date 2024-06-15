import numpy as np

from train import train
from test import test
from plot import plot_and_save_logs

np.random.seed(0)

max_steps = 10

n_learners, player_types, agents, world, folder_path = train(max_steps, plot = True)
rewards_log, actions_log = test(agents, world, player_types, max_steps)
plot_and_save_logs(player_types, rewards_log, actions_log, folder_path)