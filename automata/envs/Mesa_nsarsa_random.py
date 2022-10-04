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
import argparse 

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

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--gamma', type=float, default=0.99, help="Value of gamma for training")
    parser.add_argument('--n_steps', type=int, default=10, help="Number of steps to consider for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size (use higher if you have enough VRAM)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate of training")
    parser.add_argument('--sync_target_frames', type=int, default=1e3, help="Number of steps for synchronization between both networks")
    parser.add_argument('--episodes', type=int, default=1000, help="Learning rate of training")
    parser.add_argument('--eps_decay', type=int, default=15000)
    parser.add_argument('--gpu', type=int, default=0, help="Specify which GPU to use")
    parser.add_argument('--case', type=int, default=9)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    GAMMA = args.gamma
    BATCH_SIZE = args.batch_size
    REPLAY_SIZE = 10000
    LEARNING_RATE = args.lr
    REPLAY_START_SIZE = 1000
    SYNC_TARGET_FRAMES = args.sync_target_frames
    N_STEPS = args.n_steps

    EPSILON_START = 1.0
    EPSILON_FINAL = 0.01
    EPSILON_DECAY_LAST_STEP = args.eps_decay
    EPISODES = args.episodes
    CASE = args.case

    os.chdir('/home/kallilzie/automata_gym_cont/automata/envs')

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    env = gym.make('automata:automata-v0')
    

    directory = "action_frequency"
    dataname = "random"
    writer = SummaryWriter(comment=f"/DQN:dataname={dataname}.{CASE}_n={N_STEPS}_lr={LEARNING_RATE}_gamma={GAMMA}_epsDecay={EPSILON_DECAY_LAST_STEP}")

    for scenario in range(1,101):
        
        actions = {}
        with open('./testes_mesa/'+dataname+f'/{scenario}/case1.csv', newline='') as file:
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
        
        q_table = np.zeros([num_states, num_actions], dtype = np.float32)


        print("STARTING TRAINING!")
        epsilon = 1
        n=N_STEPS
        for episode in range(EPISODES):
            print(f"Episode: {episode}")
            A,S,R=[],[],[]
            state = env.reset()
            S.append(state)
            done = False
            T=1000
            t=0
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
                        print(f"Case {scenario}: Episode ended with reward {sum(R)}")
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
        episodes = EPISODES
        n=N_STEPS
        reward_arr = np.empty(shape=episodes)
        for episode in range(episodes):
            print(f"Episode: {episode}")
            A,S,R=[],[],[]
            state = env.reset()
            S.append(state)
            done = False
            T=1000
            t=0
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
                        print(f"Episode ended with reward {sum(R)}")
                        reward_arr[scenario-1] = sum(R)
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
        writer.add_scalar("mean_reward_scenario", np.mean(reward_arr), scenario)
                    
                        
                
            
            
            
    
# %%
