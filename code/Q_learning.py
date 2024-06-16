from collections import namedtuple
import numpy as np
from buffers import ReplayBuffer

class Q_RL:
    def __init__(
        self,
        n_history,
        gamma = 0.99,
        capacity = 500,
        alpha_mode = "Adaptive",
        epsilon = 1.,
        epsilon_decay = 0.9999, 
        mode = "Inference",
        exploration = "Uniform"
    ):
        self.gamma = gamma
        self.alpha_mode = alpha_mode
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.mode = mode
        self.exploration = exploration

        Q_shape = [2]*n_history + [2]
        self.Q = 3 / (1 - gamma) + np.random.randn(*Q_shape) - 1
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "next_state"]
        )

        self.memory = ReplayBuffer(capacity, self.experience)
        self.num_decays = 1
        self.count = np.zeros(tuple(Q_shape))
        self.rounds = 0

    def get_action(self, state):
        if self.mode == "Training" and np.random.uniform(0, 1) < self.epsilon:
            if self.exploration == "Uniform":
                return np.random.choice(2) # D = 0, C = 1
            counts = self.count[tuple(state)]+1
            p = counts[0]/np.sum(counts)
            if np.random.uniform(0, 1) < 1-p:
                return 0
            return 1
        return np.argmax(self.Q[tuple(state)])

    def reset(self):
        if self.mode == "Training":
            self.rounds += 1
            self.epsilon *= self.epsilon_decay
            self.memory.empty()

    def train(self):
        states, actions, rewards, done, next_states = self.memory.replay()

        [actions, done, rewards] = [np.reshape(x, [-1]) for x in [actions, done, rewards]]

        mapping = {
            0: "D",
            1: "C",
        }
        for state, action, reward, next_state in zip(states, actions, rewards, next_states):

            # print(f"State: {state}, Next State: {next_state}, Action: {action}, Reward: {reward}")
            # Q_ = self.Q.copy()

            old_index = tuple(np.hstack([state, action]))

            if self.alpha_mode == "Adaptive":
                alpha = 1/(self.count[old_index]+1)
            elif self.alpha_mode == "Armonic":
                alpha = 1/self.rounds
            else:
                alpha = 0.2
            
            old_Q = self.Q[old_index]
            Q_max = np.max(self.Q[tuple(next_state)])
            new_Q = (1-alpha) * old_Q + alpha*(reward + self.gamma * Q_max)
            self.Q[old_index] = new_Q

            self.count[old_index] += 1

            # print("Learned Q")
            # for index, element in np.ndenumerate(self.Q-Q_):
            #     index = "".join([mapping.get(x, x) for x in list(index)])
            #     print(f"Case {index}: {element}")
