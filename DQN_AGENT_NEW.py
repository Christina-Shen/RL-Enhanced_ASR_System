# -----------------------DDQN learning---------------------------


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        state = state.to(self.device)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.relu(self.fc3(x))

# wave DDQN
# 定义回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        super(ReplayBuffer, self).__init__()
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        #self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

class DQNAgent:
    def __init__(self, state_size, action_size,n_action):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_action=n_action
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = QNetwork(state_size, action_size,42).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size,42).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.batch_size = 64
        self.memory = ReplayBuffer(buffer_size=2000,batch_size=self.batch_size)
        self.batch_size = 64
       
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_every = 4
        self.tau = 1e-3
        self.gamma = 0.99
        self.t_step = 0
        

    def step(self, state, action, reward, next_state):
        self.memory.add(state, action, reward, next_state)
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
    
    def act(self, state):
      
        state = state.float().to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        action_values_np = action_values.cpu().data.numpy()
        _, top_actions = action_values.topk(self.n_action, dim=1)
        return top_actions
        # # Epsilon-greedy action selection
        # if random.random() > self.epsilon:
        #     return top_actions
        # else:
            

    def learn(self):
        states, actions, rewards, next_states = self.memory.sample()

        Q_values_next = self.qnetwork_target(next_states).detach()

        Q_values_next_sorted, _ = Q_values_next.sort(dim=1, descending=True)
        Q_targets_next = Q_values_next_sorted[:, :self.n_action].mean(dim=1, keepdim=True)

        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (self.gamma * Q_targets_next)
        Q_expected=0 
        for i in range(self.n_action):
            Q_expected +=(1/self.n_action)* self.qnetwork_local(states).gather(1, actions[:,i])

        loss = self.criterion(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)






