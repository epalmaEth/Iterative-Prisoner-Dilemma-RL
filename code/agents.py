import numpy as np

from Q_learning import Q_RL

class Fixed_Agent():

    def __init__(self, type) -> None:
        self.type = type
        self.k = 0
        if self.type == "Grim":
            self.remember = False
        
    def reset(self, opponent):
        self.opponent = opponent
        self.k = 0

    def response(self, opponent_state):
        self.k += 1
        if self.type == "Cu":
            return 1
        if self.type == "Du":
            return 0
        if self.type == "Random":
            return np.random.choice(2)
        if self.type == "TFT":
            return self.TFT_response(opponent_state)
        if self.type == "STFT":
            return self.TFT_response(opponent_state, 0)
        if self.type == "TF2T":
            return self.TF2T_response(opponent_state)
        return self.TFT_response(opponent_state)
    
    def TFT_response(self, opponent_state, start = 1):
        return opponent_state[-1] if self.k > 1 else start
    
    def TF2T_response(self, opponent_state):
        return 0 if opponent_state[-1] == 0 and opponent_state[-2] == 0 else 1

class RL_Agent():

    def __init__(self, i, n_learners, n_history, max_steps) -> None:
        # self.brain = Q_RL(n_history, alpha_mode="Adaptive", exploration="Proportional")
        self.brain = Q_RL(n_history, alpha_mode="Armonic", exploration="Proportional")
        # self.brain = Q_RL(n_history, alpha_mode="Fixed", exploration="Proportional")
        # self.brain = Q_RL(n_history, alpha_mode="Adaptive", exploration="Uniform")
        # self.brain = Q_RL(n_history, alpha_mode="Armonic", exploration="Uniform")
        # self.brain = Q_RL(n_history, alpha_mode="Fixed", exploration="Uniform")
        self.type = f"RL{i+1}" if n_learners > 1 else "RL"

    def reset(self, opponent):
        self.opponent = opponent
        self.brain.reset()

    def response(self, opponent_state):
        return self.brain.get_action(opponent_state)
    
    def train(self):
        self.brain.train()
