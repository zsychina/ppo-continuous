import gymnasium as gym
from agent import Agent


env = gym.make("BipedalWalker-v3", render_mode='human')

agent = Agent(
    state_dim=env.observation_space.shape[0],
    hidden_dim=128,
    action_dim=env.action_space.shape[0],
    action_highs=env.action_space.high,
    action_lows=env.action_space.low,
    device='cpu',
)

agent.load()

for episode_i in range(500):
    state, info = env.reset()
    episode_return = 0
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            done = True
        
        episode_return += reward
        
        state = next_state   
        
    print(f'{episode_i=} {episode_return=}')


agent.save()

env.close()
