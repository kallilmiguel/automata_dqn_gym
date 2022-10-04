#%%
import gym
import random
import os
import time
import torch 
import torch.optim as optim
import torch.nn as nn
import numpy as np
import csv
import pandas as pd
from lib import agent_buffer
from lib import dqn_model
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import ptan
from tensorboardX import SummaryWriter

#define global parameters
GAMMA = 0.9
BATCH_SIZE = 128
REPLAY_SIZE = 10000
LEARNING_RATE = 1E-4
REPLAY_START_SIZE = 1000
SYNC_TARGET_FRAMES = 10000
N_STEPS = 4

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_STEP = 15000
EPISODES = 500

os.chdir('/home/kallilzie/automata_gym_cont/automata/envs')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

env = gym.make('automata:automata-v0')

directory = "action_frequency"
dataname = "approve-A-10"

actions = {}
with open('./testes_mesa/'+dataname+'/case1.csv', newline='') as file:
        data = list(csv.reader(file))
for i, name in enumerate(data[0]):
    actions[name] = i 
last_actions = ['bad_A', 'bad_B', 'redo_A', 'redo_B', 'good_A', 'good_B']

cases = 1
mean_reward_episodes = [0 for i in range(cases)]
 
reward = list(map(int, data[1]))
probabilities = list(map(float, data[2]))


env.reset("SM/Renault_mesa.xml", rewards=reward, stop_crit=1, last_action=itemgetter(*last_actions)(actions), products=10, probs=probabilities)

num_actions = env.action_space.n
num_states = env.observation_space.n

net = dqn_model.DeepQNetwork2(lr=LEARNING_RATE, input_dims=num_states, emb_dims=6, fc1_dims=256, fc2_dims=256, n_actions=len(actions), device=device)
tgt_net = dqn_model.DeepQNetwork2(lr=LEARNING_RATE, input_dims=num_states, emb_dims=6, fc1_dims=256, fc2_dims=256, n_actions=len(actions), device=device)
tgt_net.load_state_dict(net.state_dict())

writer = SummaryWriter(comment=f"/DQN:dataname={dataname}_n={N_STEPS}_lr={LEARNING_RATE}_gamma={GAMMA}")


buffer = agent_buffer.ExperienceBuffer(REPLAY_SIZE, gamma = GAMMA, n_steps=N_STEPS)
agent = agent_buffer.Agent(env, buffer, reward_max=max(reward))

optimizer = optim.RMSprop(net.parameters(), lr =LEARNING_RATE)

#%%

episode_idx = 0
step_idx = 0
episodes = EPISODES
ts = time.time()
best_total_reward = None
for i in range(episodes):
    total_reward = 0
    print(f"Episode: {i}")
    observation = agent._reset()
    while True:
        step_idx+=1
        epsilon = max(EPSILON_FINAL, EPSILON_START - step_idx / EPSILON_DECAY_LAST_STEP)
        reward, done = agent.play_step(net, epsilon)
        total_reward += reward
        if not done:
            writer.add_scalar("epsilon", epsilon, step_idx)
            
        else:
            print(f"Episode ended with reward {total_reward}")
            writer.add_scalar("total_reward", total_reward, i)
            if best_total_reward is None or best_total_reward < total_reward:
                if best_total_reward is not None:
                    print("Best reward updated %.3f -> %.3f" %(best_total_reward, total_reward))
                best_total_reward = total_reward
            break
        
        if len(buffer) < REPLAY_START_SIZE:
            continue
        
        if step_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = agent.calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        for param in net.parameters():
            param.grad.data.clamp_(-1,1)
        optimizer.step()
        writer.add_scalar("loss", loss_t, step_idx)
writer.close()
            
        
        
            
            
    
# %%
