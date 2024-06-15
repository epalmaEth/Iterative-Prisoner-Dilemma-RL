from world import World
from agents import RL_Agent, Fixed_Agent

def create_game(n_learners, fixed_players, n_history, max_steps, prob_return):
    # Create agents
    agents = [RL_Agent(i, n_learners, n_history, max_steps) for i in range(n_learners)]
    for type, n_agents in fixed_players.items():
        for _ in range(n_agents):
            agents.append(Fixed_Agent(type=type))
    if prob_return:
        world = World(agents, n_history, max_steps, noise_prob=0.1)
    else:
        world = World(agents, n_history, max_steps)
    return agents, world