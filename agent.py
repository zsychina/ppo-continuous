import torch
from torch.distributions import MultivariateNormal

import numpy as np
from mlp import MLP


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]
        
class Agent:
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 action_highs,
                 action_lows,
                 action_std_init=0.6,
                 action_std_decay=0.05,
                 min_action_std=0.1,
                 action_std_decay_period=5e4,
                 lmbda=0.95,
                 gamma=0.99,
                 eps_clip=0.2,
                 K_epochs=10,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 device='cpu',
                 model_path='./checkpoint/'):
        
        self.timestep = 0
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.action_highs = action_highs
        self.action_lows = action_lows
        
        self.action_std = action_std_init
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        self.action_std_decay=action_std_decay
        self.min_action_std=min_action_std
        self.action_std_decay_period=action_std_decay_period
        
        self.lmbda = lmbda
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr_actor = lr_actor  
        self.lr_critic = lr_critic
        self.buffer = RolloutBuffer()
        self.device = device
        self.model_path = model_path
        
        self.actor = MLP(state_dim=state_dim, hidden_dim=hidden_dim, out_dim=action_dim, actor=True).to(device)
        self.critic = MLP(state_dim=state_dim, hidden_dim=hidden_dim, out_dim=1, actor=False).to(device)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        self.mse = torch.nn.MSELoss()
        
    def set_action_var(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
        self.set_action_var(self.action_std)

    def action_denorm(self, action_norm: list):
        amp = (np.array(self.action_highs) - np.array(self.action_lows)) / 2.0 # for tanh(inf)-tanh(-inf)=2
        bias = (np.array(self.action_highs) + np.array(self.action_lows)) / 2.0
        return (amp * action_norm + bias).tolist()

    def take_action(self, state):
        self.timestep += 1
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = state.unsqueeze(0)
        # [1, action_dim]
        action_mean = self.actor(state)
        # [1, action_dim, action_dim]
        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        dist = MultivariateNormal(action_mean, cov_mat)
        # [1, action_dim]
        action = dist.sample()
        action = action.squeeze()
        
        if self.timestep % self.action_std_decay_period == 0:
            self.decay_action_std(self.action_std_decay, self.min_action_std)
        return self.action_denorm(action.tolist())
    
    def eval(self, state):
        pass
   
        
    def update(self):
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float).reshape(-1, self.state_dim).to(self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.int64).reshape(-1, self.action_dim).to(self.device)
        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float).reshape(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(self.buffer.next_states), dtype=torch.float).reshape(-1, self.state_dim).to(self.device)
        dones = torch.tensor(np.array(self.buffer.dones), dtype=torch.float).reshape(-1, 1).to(self.device)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        
        advantage = 0
        advantage_list = []
        td_delta = td_delta.detach().cpu().numpy()
        for delta in td_delta[::-1]:
            advantage = delta + self.gamma * self.lmbda * advantage
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)
        
        # old_log_probs = torch.log(self.actor(states).gather(-1, actions)).detach()
        # 对于连续动作，使用概率密度函数
        old_action_means = self.actor(states)
        old_cov_mat = torch.diag(self.action_var).unsqueeze(0)
        old_dist = MultivariateNormal(old_action_means.detach(), old_cov_mat)
        old_log_probs = old_dist.log_prob(actions)
        
        for _ in range(self.K_epochs):
            # log_probs = torch.log(self.actor(states).gather(-1, actions))
            # 对于连续动作，使用概率密度函数
            action_mean = self.actor(states)
            cov_mat = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action_mean, cov_mat)
            log_probs = dist.log_prob(actions)            
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage_list
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage_list
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                self.mse(self.critic(states), td_target.detach())
            )
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

        self.buffer.clear()
        
    def save(self):
        torch.save(self.actor.state_dict(), self.model_path+'actor.pt')
        torch.save(self.critic.state_dict(), self.model_path+'critic.pt')
    
    def load(self):
        self.actor.load_state_dict(torch.load(self.model_path+'actor.pt', map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(self.model_path+'critic.pt', map_location=lambda storage, loc: storage))
    
