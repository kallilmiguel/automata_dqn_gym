
from re import M
import time
import numpy as np
import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'next_state']
)

class ExperienceBuffer:
    def __init__(self, capacity, gamma, n_steps=2):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = collections.deque(maxlen=capacity)
        self.steps_until_last_done = 0
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
        
        if self.__len__() >= self.n_steps and self.n_steps > 1 and self.buffer[-self.n_steps]['done'] == False:
            for i in range(self.n_steps-1):
                self.buffer[-self.n_steps]['reward']+= self.gamma**(i+1)*self.buffer[-self.n_steps+i+1]['reward']
                last_state = self.buffer[-self.n_steps+i+1]['state']
                if(self.buffer[-self.n_steps+i+1]['done']):
                    break
            self.buffer[-self.n_steps]['next_state'] = last_state
                
    
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx].values() for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),\
            np.array(dones), np.array(next_states)
            
            
class Agent:
    def __init__(self, env, exp_buffer, reward_max):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.reward_scale = (2)/(2*reward_max) #multiply with this scale to clip all rewards between -1 and 1 
        
    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0
        
    def play_step(self, net, epsilon=0.0):
        
        action = self.choose_action(self.state, net, epsilon)
            
        # do step in the environment
        new_state, reward, is_done, _ =  self.env.step(action)
        self.total_reward += reward
        
        exp = {
            'state': self.state, 
            'action': action, 
            'reward': reward, 
            'done': is_done, 
            'next_state': new_state
        }
        self.exp_buffer.append(exp)
        self.state = new_state
        
        return reward, is_done
    
    @torch.no_grad()
    def play_step_usual(self, net, epsilon=0.0, device="cpu"):
        
        if np.random.uniform(0,1) < epsilon:
            action = self.env.action_space.sample()
        else:
            observation = torch.tensor(self.state).to(device)
            q_vals = net(observation).squeeze()
            action = torch.argmax(q_vals).item()
        
            
        # do step in the environment
        new_state, reward, is_done, _ =  self.env.step(action)
        self.total_reward += reward
        
        exp = {
            'state': self.state, 
            'action': action, 
            'reward': reward, 
            'done': is_done, 
            'next_state': new_state
        }
        self.exp_buffer.append(exp)
        self.state = new_state
        
        return reward, is_done
    
    @torch.no_grad()
    def choose_action(self, observation, net, epsilon):
        
        pt = np.array(self.env.possible_transitions())
        uncontrollable = np.array(self.env.ncontrollable)
        ptu = np.intersect1d(pt,uncontrollable)
        probs = np.array([[ptu[i],self.env.probs[ptu[i]]] for i in range(len(ptu)) if self.env.probs[ptu[i]]>0])
        controllable = np.array(self.env.controllable)
        ptc = np.intersect1d(pt,controllable)
        
        actions = np.array([])
        if(probs.size>0):
            actions = np.append(actions, random.choices(probs[:,0],weights=probs[:,1]))

        
        if(len(probs)>0):
            for i in range(len(probs)):
                ptu = np.delete(ptu, np.where(ptu==probs[i,0]))
        if(ptu.size>0):
            actions=np.append(actions, ptu)
        
        if(ptc.size>0):
            if np.random.uniform(0,1) < epsilon:
                actions = np.append(actions, ptc[np.random.randint(0,ptc.size)])
            else:
                state = torch.tensor([observation]).to(net.device)
                q_vals = net(state).squeeze()
                actions = np.append(actions,ptc[torch.argmax(q_vals[ptc]).item()])
        
        return actions[np.random.randint(0,actions.size)].astype(np.int32)
    
    def calc_loss(self, batch, net, tgt_net, device="cpu"):
        states, actions, rewards, dones, next_states = batch
        
        states_v = torch.tensor(np.array(states, copy=False)).to(device)
        next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
        actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_v = torch.tensor(rewards).to(device) * self.reward_scale
        done_mask = torch.BoolTensor(dones).to(device)
        
        state_action_values = net(states_v).gather(
            1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_actions = net(next_states_v).max(1)[1]
            next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()       
            
        expected_state_action_values = next_state_values * self.exp_buffer.gamma**self.exp_buffer.n_steps +\
            rewards_v
            
        return nn.MSELoss()(state_action_values, expected_state_action_values)
        
            