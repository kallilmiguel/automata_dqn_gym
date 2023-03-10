# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:25:12 2021

@author: Lucas
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class DeepQNetwork(nn.Module): #aqui tá dando ruim, verificar
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)    #(1, 256)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)      #(256, 256)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)     #(256, 26)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)
    
    def forward(self, state):
        #state = state.float().reshape((-1,1)) # essa linha é importantíssima
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions
    
class DeepQNetwork2(nn.Module): #aqui tá dando ruim, verificar
    def __init__(self, lr, input_dims, emb_dims, fc1_dims, fc2_dims, n_actions, device):
        super(DeepQNetwork2, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.emb_dims = emb_dims
        self.device = device
        self.emb = nn.Embedding(self.input_dims, self.emb_dims).to(self.device)
        self.fc1 = nn.Linear(self.emb_dims, self.fc1_dims).to(self.device)  
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims).to(self.device)    
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions).to(self.device) 
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)   
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
            
    def forward(self, state):
        #state = state.float().reshape((-1,1)) # essa linha é importantíssima
        x = self.emb(state.long())
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        actions = self.fc3(x)
        
        return actions
    
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, 
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-6):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, fc1_dims=256,
                                         fc2_dims=256, n_actions=n_actions)
        self.state_memory = np.zeros((self.mem_size, input_dims), 
                                         dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), 
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
        
    def choose_action(self, observation, possible_transitions=[]):
        if np.random.random() > self.epsilon: # exploit (known action)
            state = T.tensor([observation], dtype=T.float32).to(self.Q_eval.device) 
            rewards = self.Q_eval.forward(state) 
            
            #action = T.argmax(rewards).item()
            mask = T.ones(rewards.size(), dtype=bool)
            transitions = T.tensor(range(rewards.size(0)), dtype=T.int32)
            for r in rewards:
                #action = T.argmax(rewards[action]).item()
                idx = T.argmax(rewards[mask]).item()
                action = transitions[mask][idx]
                #action = T.argmax(rewards).item()
                if action in possible_transitions:
                    return action
                mask[action] = False
            
        else:
            if(len(possible_transitions)): # explore (random action)
                action = np.random.choice([int(i) for i in possible_transitions])
            else:
                action = np.random.choice(self.action_space)
        
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                                                    else self.eps_min

    












