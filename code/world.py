import numpy as np

class World():
    def __init__(self, agents, n_history, max_steps, noise_prob = None):
        self.agents = agents   
        self.n_history = n_history    
        self.max_steps = max_steps
        self.noise_prob = noise_prob
        self.n_agents = len(agents)
        self.reward_matrix = np.array([[1, 5],
                                       [0, 3]])   
        self.state = None

    def reset(self):
        self.episode = 0

        pairings = self.pick_pairings()

        if self.state is None:
            self.state = np.ones((self.n_agents, self.n_history), dtype=int)
        else:
            new_state = self.state.copy()
            for player1, player2 in pairings.items():
                new_state[player1] = self.state[self.agents[player2].opponent]
                new_state[player2] = self.state[self.agents[player1].opponent]
            self.state = new_state.copy()

        for i, agent in enumerate(self.agents):
            agent.reset(pairings[i])

        return self.state.copy()
    
    def pick_pairings(self):
        indeces = np.arange(self.n_agents)
        available_indeces = indeces[indeces != -1]
        pairings = {}
        while len(available_indeces) > 1:
            i_player_1 = available_indeces[np.random.randint(len(available_indeces))]
            available_indeces = available_indeces[available_indeces != i_player_1]
            i_player_2 = available_indeces[np.random.randint(len(available_indeces))]
            indeces[i_player_1] = indeces[i_player_2] = -1
            available_indeces = indeces[indeces != -1]
            pairings[i_player_1] = i_player_2
            pairings[i_player_2] = i_player_1
        return pairings

    def step(self, actions):

        self.episode += 1
        done = self.episode == self.max_steps

        self.state[:, :self.n_history-1] = self.state[:, 1:self.n_history]

        for i, action in enumerate(actions):
            self.state[i, -1] = action
            
        rewards = self.compute_reward(done)

        if self.noise_prob is not None:
            if np.random.uniform(0,1) < self.noise_prob:
                rewards += 2*np.random.random(self.n_agents)-1
            
        return self.state, rewards, done 
    
    def compute_reward(self, done):
        rewards = np.zeros(self.n_agents)
        for i, agent in enumerate(self.agents):
            response_1 = self.state[i, -1]
            response_2 = self.state[agent.opponent, -1]
            rewards[i] = self.reward_matrix[response_1, response_2] # + self.reward_matrix[response_2, response_1]
        return rewards

