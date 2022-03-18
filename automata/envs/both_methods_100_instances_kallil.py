# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 19:52:24 2021

@author: Lucas
"""

#!/usr/bin/env python3


import gym
import random
import numpy as np
import time
#from policy import CustomEpsGreedyQPolicy
import matplotlib.pyplot as plt
import csv 
import pandas as pd
import seaborn as sns
import os
from DQN import Agent
#import pybullet_envs # apagaar depois
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from gym import wrappers
#from torch.autograd import Variable
#from collections import deque


os.environ['KMP_DUPLICATE_LIB_OK']='True'

start = time.time()


def choose_action(env):
    pt = np.array(env.possible_transitions())
    action = pt[env.possible_space.sample()]
    return action

def epsilon_greedy(env, epsilon, q_table, state): 
    
    pt = np.array(env.possible_transitions())
    uncontrollable = np.array(env.ncontrollable)
    ptu = np.intersect1d(pt,uncontrollable)
    probs = np.array([[ptu[i],env.probs[ptu[i]]] for i in range(len(ptu)) if env.probs[ptu[i]]>0])
    controllable = np.array(env.controllable)
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
        if random.uniform(0,1) < epsilon:
            actions = np.append(actions, ptc[random.randint(0,ptc.size-1)])
        else:
            actions = np.append(actions,ptc[np.argmax(q_table[state,ptc])])
        
    return actions[random.randint(0,actions.size-1)].astype(np.int32)
    
            

def q_possible():
    q_p = []
    mp = env.mapping(index=False)
    for i in range(len(env.possible_transitions())):
        q_p.append([mp[env.possible_transitions()[i]], q_table[env.actual_state][env.possible_transitions()[i]]])
    return q_p

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'automata-v0' in env:
        #print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]



env = gym.make('automata:automata-v0')
cases=9
mean_reward_episodes = [0 for i in range(cases)]


info_int=[]
info_rdn=[]
final_states_int=[]
final_states_rdn=[]
info_dql = []
final_states_dql=[]


bad_A = 0
bad_B = 1
good_A = 10
good_B = 11
#good_A = 12
#good_B = 13
redo_A = 12
redo_B = 13
#redo_A = 17
#redo_B = 18


directory="dados_lucas/"
dataname="random"
last_action=[bad_A, bad_B, good_A, good_B, redo_A, redo_B]
last_actions=[bad_A, bad_B, good_A, good_B, redo_A, redo_B]
xValues=[]


agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=35,
                  eps_end=0.01, input_dims=1, lr=0.003) # mudar o input dims



rewards_avg = pd.DataFrame({'sct': [], 'nsarsa': [], 'deep': []})
rewards_avg.head()
#print(rewards_avg)


reward_sum_sct = 0
reward_sum_nsarsa = 0
reward_sum_dql = 0

reward_avg_sct = 0
reward_avg_nsarsa = 0
reward_avg_dql = 0

instances = 20
for inst in range(instances):  
#for inst in range(instances):      
    reward_sum_sct = 0
    reward_sum_nsarsa = 0
    reward_sum_dql = 0
    for k in range(cases):
        print("case: ",str(k+1))
        with open('testes/'+dataname+'/'+str(inst+1)+'/case'+str(k+1)+'.csv', newline='') as csvfile:
        #with open('testes/'+dataname+'/case'+str(k+1)+'.csv', newline='') as csvfile:
            data = list(csv.reader(csvfile))
        reward = list(map(int, data[1]))
        probabilities = list(map(float, data[2]))
        
        env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
        #env.reset("SM/Renault_mesa.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
     
        #last_actions=last_action
        num_actions = env.action_space.n
        num_states = env.observation_space.n
        
        q_table = np.zeros([num_states, num_actions], dtype = np.float32)
            
        #Alterar esse valor para aparecer no eixo x do gráfico
        xValues.append(100*env.probs[0])
        
        lr = 0.1
        gamma = 0.9
        epsilon = 0.5
        
        bad=[1011]
        #bad = [457,257,912,259,304,1226,888,1220,313,672]
        
        
        
            
        ##NStepSarsa
        episodes = 3000
        n=10
        start = time.time()
        for i in range(episodes):
            steps=0
            #print("Case: {}/{} ----- Episode: {}/{}".format(k+1,cases,i,episodes))
            A,S,R=[],[],[]
            state = env.reset()
            S.append(state)
            done = False
            T=100000
            t=0
            action=epsilon_greedy(env,epsilon, q_table, state)
            A.append(action)
            while True:
                if(t<T):
                    s_n,r_n,done,_ = env.step(action)
                    S.append(s_n)
                    R.append(r_n)
                    if(done):
                        T=t+1
                        #print("---Steps:{}".format(steps))
                    else:
                        action = epsilon_greedy(env, epsilon, q_table, s_n)
                        steps+=1
                        A.append(action)
                tau=t-n+1
                if(tau>=0):
                    G=0
                    for z in range(tau+1, min(tau+n,T)+1):
                        G+=(gamma**(z-tau-1))*R[z-1]
                    if(tau+n<T):
                        G = G+(gamma**n)*q_table[S[tau+n],A[tau+n]]
                    q_table[S[tau],A[tau]] += lr*(G-q_table[S[tau],A[tau]])
                if(tau==T-1):
                    break
                t+=1
        
            
                
        
    
        #Testando decisões inteligentes
        epsilon=0
        episodes_nsarsa = 50
        
        #print("\tTeste Inteligente")
        for i in range(episodes_nsarsa):
            state = env.reset()
            total_reward=0
            #env.render()
            
            done = False
            while not done:
                
                action = epsilon_greedy(env, epsilon, q_table, state)
                #if(action in last_actions or action ==12 or action ==13):
                if(action in last_actions):
                    final_states_int.append((action,k+1))
                
    #            if(state in l and (action!=20 and action !=22)):
     #              print("State:{}/Action:{}".format(state,action))
                    
               
                next_state, reward, done, info = env.step(action)
                total_reward+=reward
                
                state = next_state
            
            info_int.append((total_reward, 10*(k+1), "Supervisory+RL")) 
            #print("nsarsa: ",total_reward)
            reward_sum_nsarsa += total_reward
            #info_int.append((total_reward, xValues[k], "Supervisory+RL"))
            #print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
            
        #Testando decisões aleatórias
        epsilon=1
        episodes_sct = 50
        rewards_rdn=[]
        #print("\tTeste Aleatório")
        for i in range(episodes_sct):
            state = env.reset()
            total_reward=0
            #env.render()
            
            done = False
            cars=0
            count=0
            
            while not done:
    
                action = epsilon_greedy(env, epsilon, q_table, state)
                
                #if(action in last_actions or action==12 or action==13):
                if(action in last_actions):
                    final_states_rdn.append((action,k+1))
    
                
                next_state, reward, done, info = env.step(action)
                total_reward+=reward
                
                state = next_state
            
            info_rdn.append((total_reward, 10*(k+1), "Supervisory"))   
            #print("sct:", total_reward)
            reward_sum_sct += total_reward
            #info_rdn.append((total_reward, xValues[k], "Supervisory"))
            #print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
            
        
        
        # DQL        
        transition_names = list(map(str, data[0]))
        reward = list(map(int, data[1]))
        probabilities = list(map(float, data[2]))
        
        env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
        #env.reset("SM/Renault_mesa.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
        
        #last_actions=last_action
        num_actions = env.action_space.n
        num_states = env.observation_space.n
        
        actions = env.possible_transitions()
        
        scores, eps_history = [], []
        episodes_dql = 100#0
        sum_reward_episodes = 0
        for i in range(episodes_dql):
            score = 0
            done = False
            observation = env.reset()
            total_reward = 0
            while not done:
                possible_transitions = env.possible_transitions()
                action = agent.choose_action(observation, possible_transitions)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
                
                if action in last_actions:
                    final_states_dql.append((action,k+1))
            
            sum_reward_episodes += score
            
            scores.append(score)
            eps_history.append(agent.epsilon)
            
            info_dql.append((score, 10*(k+1), "DQL"))   
            #print("dql: ", score)
            reward_sum_dql += score
            #info_dql.append((total_reward, xValues[k], "DQL"))
            #print("\t\tEpisode: {}, Test Case: {}, Mean Reward: {}".format(i, (k+1)*10, score))
            
            #print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
            #print('episode ', i, 'score %.2f' % score, 'epsilon %.5f' % agent.epsilon)
                  #'average score %.2f' % avg_score, 
        mean_reward_episodes[k] = sum_reward_episodes/episodes_dql
        
    reward_avg_sct = reward_sum_sct/(episodes_sct*cases)
    reward_avg_nsarsa = reward_sum_nsarsa/(episodes_nsarsa*cases)
    reward_avg_dql = reward_sum_dql/(episodes_dql*cases)
    print("média sct: ", reward_avg_sct)
    print("média nsarsa: ", reward_avg_nsarsa)
    print("média dql: ", reward_avg_dql)
    print("-----------------------")
    
    rewards_line = pd.DataFrame({'sct': [reward_avg_sct], 'nsarsa': [reward_avg_nsarsa], 'deep': [reward_avg_dql]})
    rewards_avg = rewards_avg.append(rewards_line)
    #print(rewards_avg)
    
        
        #info_dql.append((mean_reward_episodes[k], 10*k, "DQL"))    
       
        
        
    #rewards_avg = pd.DataFrame({'sct': [], 'nsarsa': [], 'deep': []})
    
    
    
    
#print(rewards_avg)


#path = "C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym_cont/automata/envs/dados_lucas_100/out4.csv"

#rewards_avg.to_csv(path_or_buf=path)
    
    
    
