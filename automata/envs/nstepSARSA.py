#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 23:02:58 2020

@author: kallil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:49:07 2020

@author: kallil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:22:34 2020

@author: kallil
"""
#%%
import gym
import random
import numpy as np
import time
import csv 
from operator import itemgetter
from tensorboardX import SummaryWriter

GAMMA = 0.9
BATCH_SIZE = 128
REPLAY_SIZE = 10000
LEARNING_RATE = 1E-4
N_STEPS = 50
EPSILON = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY_LAST_EP = 2000

case=9
last_actions = [0,1,10,11]

def q_possible():
    q_p = []
    mp = env.mapping(index=False)
    for i in range(len(env.possible_transitions())):
        q_p.append([mp[env.possible_transitions()[i]], q_table[env.actual_state][env.possible_transitions()[i]]])
    return q_p

env = gym.make('automata:automata-v0')


directory = "action_frequency"
dataname = "approve-A-90"

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
writer = SummaryWriter(comment=f"/NSARSA_dataname={dataname}_n={N_STEPS}_lr={LEARNING_RATE}_gamma={GAMMA}")


num_actions = env.action_space.n
num_states = env.observation_space.n


q_table = np.zeros([num_states, num_actions], dtype = np.float32)


def choose_action(env):
    pt = np.array(env.possible_transitions())
    action = pt[env.possible_space.sample()]
    return action

def epsilon_greedy(env, epsilon, q_table, state): 
    action = -1
    pt = np.array(env.possible_transitions())
    uncontrollable = np.array(env.ncontrollable)
    ptu = np.intersect1d(pt,uncontrollable)
    probs = [[env.probs[ptu[i]], ptu[i]] for i in range(len(ptu)) if env.probs[ptu[i]]>0]
    
    if(len(probs)>0):
        for i in range(len(probs)):
            pt = np.delete(pt, np.where(pt==ptu[i]))
    
    while True:
    
        
        random.shuffle(probs)
        
        for i in range(len(probs)):
            if(np.random.uniform(0,1)<probs[i][0]):
                return probs[i][1]
                
        if(pt.size>0):
            controllable = np.array(env.controllable)
            ptc = np.intersect1d(pt,controllable)   
        
            if ptc.size>0:
                if random.uniform(0,1) < epsilon:
                    action = pt[random.randint(0,pt.size-1)]
                else:
                    action = ptc[np.argmax(q_table[state,ptc])]
            else: 
                    action = pt[random.randint(0,pt.size-1)]
                
        if action !=-1:
            break
        
    return action
#%%          
##NStepSarsa
print("STARTING TRAINING!")
epsilon = 1
episodes = 2000
n=N_STEPS
for episode in range(episodes):
    print(f"Episode: {episode}")
    A,S,R=[],[],[]
    state = env.reset()
    S.append(state)
    done = False
    T=1000
    t=0
    epsilon = max(EPSILON_FINAL, EPSILON - episode / EPSILON_DECAY_LAST_EP)
    action=epsilon_greedy(env,epsilon, q_table, state)
    A.append(action)
    total_reward = 0
    while True:
        if(t<T):
            s_n,r_n,done,_ = env.step(action)
            S.append(s_n)
            R.append(r_n)
            total_reward += r_n
            if(done):
                writer.add_scalar("total_reward_train", total_reward, episode)
                print(f"Episode ended with reward {sum(R)}")
                T=t+1
            else:
                action = epsilon_greedy(env, epsilon, q_table, state)
                A.append(action)
        tau = t-n+1
        if(tau>=0):
            G=0
            for i in range(tau+1, min(tau+n,T)):
                G+=(GAMMA**(i-tau-1))*R[i]
            if(tau+n<T):
                G = G+(GAMMA**n)*q_table[S[tau+n],A[tau+n]]
            q_table[S[tau],A[tau]] += LEARNING_RATE*(G-q_table[S[tau],A[tau]])
     
            
        if(tau==T-1):
            break
        t+=1
  
print("NOW STARTING TEST PHASE!!")      
epsilon = 0
episodes = 2000
n=N_STEPS
for episode in range(episodes):
    print(f"Episode: {episode}")
    A,S,R=[],[],[]
    state = env.reset()
    S.append(state)
    done = False
    T=1000
    t=0
    epsilon = max(EPSILON_FINAL, EPSILON - episode / EPSILON_DECAY_LAST_EP)
    action=epsilon_greedy(env,epsilon, q_table, state)
    A.append(action)
    total_reward = 0
    while True:
        if(t<T):
            s_n,r_n,done,_ = env.step(action)
            S.append(s_n)
            R.append(r_n)
            total_reward += r_n
            if(done):
                writer.add_scalar("total_reward_test", total_reward, episode)
                print(f"Episode ended with reward {sum(R)}")
                T=t+1
            else:
                action = epsilon_greedy(env, epsilon, q_table, state)
                A.append(action)
        tau = t-n+1
        if(tau>=0):
            G=0
            for i in range(tau+1, min(tau+n,T)):
                G+=(GAMMA**(i-tau-1))*R[i]
            if(tau+n<T):
                G = G+(GAMMA**n)*q_table[S[tau+n],A[tau+n]]
            q_table[S[tau],A[tau]] += LEARNING_RATE*(G-q_table[S[tau],A[tau]])
     
            
        if(tau==T-1):
            break
        t+=1


# %%
