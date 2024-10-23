import gymnasium as gym
from agent import Agent
import matplotlib.pyplot as plt

env = gym.make("BipedalWalker-v3")
# env = gym.make('Pendulum-v1')

agent = Agent(
    state_dim=env.observation_space.shape[0],
    hidden_dim=128,
    action_dim=env.action_space.shape[0],
    action_highs=env.action_space.high,
    action_lows=env.action_space.low,
    device='cpu',
)

# agent.load()

reward_step = []
reward_episode = []
for episode_i in range(5000):
    state, info = env.reset()

    episode_return = 0

    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            done = True
            
        agent.buffer.states.append(state)
        agent.buffer.actions.append(action)
        agent.buffer.rewards.append(reward)
        agent.buffer.next_states.append(next_state)
        agent.buffer.dones.append(done)
        
        state = next_state   
        
        episode_return += reward
        reward_step.append(reward)     
        
    agent.update()
    if episode_i % 100 == 0:
        agent.save()
    
    print(f'{episode_i=} {episode_return=}')
    reward_episode.append(episode_return)

agent.save()

plt.plot(reward_step)
plt.show()


env.close()
    
    