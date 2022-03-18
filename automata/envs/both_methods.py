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
        print("Remove {} from registry".format(env))
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
dataname="bad_A_cost"
last_action=[bad_A, bad_B, good_A, good_B, redo_A, redo_B]
last_actions=[bad_A, bad_B, good_A, good_B, redo_A, redo_B]
xValues=[]


agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=35,
                  eps_end=0.01, input_dims=1, lr=0.003) # mudar o input dims





for k in range(cases):

    #with open('testes_mesa/'+dataname+'/case'+str(k+1)+'.csv', newline='') as csvfile:
    with open('testes/'+dataname+'/case'+str(k+1)+'.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    reward = list(map(int, data[1]))
    probabilities = list(map(float, data[2]))
    
    env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    #env.reset("SM/Renault_mesa.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    #end = time.time()
    
    #print(end-start)
    
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
        print("Case: {}/{} ----- Episode: {}/{}".format(k+1,cases,i,episodes))
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
                    print("---Steps:{}".format(steps))
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
    episodes = 50
    
    print("\tTeste Inteligente")
    for i in range(episodes):
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
        #info_int.append((total_reward, xValues[k], "Supervisory+RL"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
        
    #Testando decisões aleatórias
    epsilon=1
    episodes = 50
    rewards_rdn=[]
    print("\tTeste Aleatório")
    for i in range(episodes):
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
        #info_rdn.append((total_reward, xValues[k], "Supervisory"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
        
    
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
    episodes = 100#0
    sum_reward_episodes = 0
    for i in range(episodes):
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
        #info_dql.append((total_reward, xValues[k], "DQL"))
        print("\t\tEpisode: {}, Test Case: {}, Mean Reward: {}".format(i, (k+1)*10, score))
        
        #print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
        #print('episode ', i, 'score %.2f' % score, 'epsilon %.5f' % agent.epsilon)
              #'average score %.2f' % avg_score, 
    mean_reward_episodes[k] = sum_reward_episodes/episodes
    
    #info_dql.append((mean_reward_episodes[k], 10*k, "DQL"))    
   
        
        
    

# Alterar dataname para salvar diferentes bases de dados
reward_dataname=directory+dataname+"_reward.csv"
occurrences_int_dataname=directory+dataname+"_fsInt.csv"
occurrences_dql_dataname=directory+dataname+"_fsDQL.csv"
occurrences_rnd_dataname=directory+dataname+"_fsRnd.csv"

occurrences_redo_int = directory+dataname+"_redo.csv"


#occurrences_redo_dql = directory+dataname+"_redo.csv"


#nome do eixo x do gráfico
xlabel_name="xl"

fsInt=[]
fsRdn=[]
fsdql=[]
redo=[]
#print(last_actions)


for i in last_actions:
    for j in range(cases):
        fsInt.append((final_states_int.count((i,j+1)), (j+1)*10, env.mapping()[i][1]))
        fsRdn.append((final_states_rdn.count((i,j+1)), (j+1)*10, env.mapping()[i][1]))
        fsdql.append((final_states_dql.count((i,j+1)), (j+1)*10, env.mapping()[i][1]))
        if i==redo_A or i==redo_B:
            redo.append((final_states_int.count((i,j+1)),(j+1)*10,env.mapping()[i][1],"Supervisory+RL"))
            redo.append((final_states_rdn.count((i,j+1)),(j+1)*10,env.mapping()[i][1],"Supervisory")) 
            redo.append((final_states_dql.count((i,j+1)),(j+1)*10,env.mapping()[i][1],"DQL"))


print(redo)


        
list2Int=[]
list2Rdn=[]
list2dql=[]
redoInt=[]
redoRdn=[]
redoDqn=[]


for i in range(0,9):
    list2Int.append((fsInt[i][1],fsInt[i][0], fsInt[i+9][0],fsInt[i+18][0],fsInt[i+27][0]))
    redoInt.append((fsInt[i][1],fsInt[i+36][0],fsInt[i+45][0]))


for i in range(0,9):
    list2Rdn.append((fsRdn[i][1],fsRdn[i][0], fsRdn[i+9][0],fsRdn[i+18][0],fsRdn[i+27][0]))
    redoRdn.append((fsRdn[i][1],fsRdn[i+36][0],fsRdn[i+45][0]))
    
for i in range(0,9):
    list2dql.append((fsdql[i][1],fsdql[i][0], fsdql[i+9][0],fsdql[i+18][0],fsdql[i+27][0]))
    redoDqn.append((fsdql[i][1],fsdql[i+36][0],fsdql[i+45][0]))    


#print(redoDqn)
#print(redoRdn)
#print(redoInt)
#print(redo)

data = np.vstack((info_int, info_rdn, info_dql))
data = pd.DataFrame(data, columns=["yl",  xlabel_name, "method"])

states_dql = pd.DataFrame(list2dql,columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
states_int = pd.DataFrame(list2Int,columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
states_rdn = pd.DataFrame(list2Rdn, columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
redo = pd.DataFrame(redo, columns=["Number of Occurrences", xlabel_name, "Event", "Method"])
#redoRdn = pd.DataFrame(redoRdn, columns=[xlabel_name,"Rework Type 1","Rework Type 2"])
#print(redo)


data.to_csv(reward_dataname)
states_int.to_csv(occurrences_int_dataname)
states_dql.to_csv(occurrences_dql_dataname)
states_rdn.to_csv(occurrences_rnd_dataname)
redo.to_csv(occurrences_redo_int)
#redo.to_csv(occurrences_redo_dql)
# redoRdn.to_csv(occurrences_redo_rdn)
#print(redo)


intel = pd.read_csv(occurrences_int_dataname)
dql = pd.read_csv(occurrences_dql_dataname)
randomic = pd.read_csv(occurrences_rnd_dataname)
df = pd.read_csv(reward_dataname)
redo = pd.read_csv(occurrences_redo_int)
#redo = pd.read_csv(occurrences_redo_dql)
# redoRdn = pd.read_csv(occurrences_redo_rdn)
print(redo)


intel = intel.drop(["Unnamed: 0"],axis=1)
dql = dql.drop(["Unnamed: 0"],axis=1)
randomic = randomic.drop(["Unnamed: 0"], axis=1)
redo = redo.drop(["Unnamed: 0"], axis=1)
# redoRdn = redoRdn.drop(["Unnamed: 0"], axis=1)
print(redo)

pd.set_option("display.max_rows", 100, "display.max_columns", 100)
df.head(1000)





# PLOT DAS RECOMPENSAS
#plot = sns.lineplot(data=df,  hue='method', x='xl', y='yl')
sns.lineplot(data=df,  hue='method', x='xl', y='yl', err_style=None)
plt.savefig('Plots/Both/'+dataname+'/rewards-'+dataname+'.eps', dpi=300)

#plt.savefig('Plots/Sarsa/'+dataname+'/rewards-'+dataname+'.eps', dpi=300)
#plt.savefig('Plots_mesa/Both/'+dataname+'/rewards-'+dataname+'.eps', dpi=300)




# PLOTS DAS AÇÕES
plot = intel.plot.bar(x=xlabel_name, stacked=True)
plot.set_xlabel(xlabel_name)
plot.set_ylabel("Número de Ocorrências")
plt.savefig('Plots/Both/'+dataname+'/sarsa-actionstaken-'+dataname+'.eps', dpi=300)
#plt.savefig('Plots_mesa/Sarsa/'+dataname+'/sarsa-actionstaken-'+dataname+'.eps', dpi=300)


plot = randomic.plot.bar(x=xlabel_name, stacked=True)
plot.set_xlabel(xlabel_name)
plot.set_ylabel("Número de Ocorrências")
plt.savefig('Plots/Both/'+dataname+'/SCT-actionstaken-'+dataname+'.eps', dpi=300)
#plt.savefig('Plots_mesa/Sarsa/'+dataname+'/SCT-actionstaken-'+dataname+'.eps', dpi=300)


plot = dql.plot.bar(x=xlabel_name, stacked=True)
plot.set_xlabel(xlabel_name)
plot.set_ylabel("Número de Ocorrências")
plt.savefig('Plots/Both/'+dataname+'/DQL-actionstaken-'+dataname+'.eps', dpi=300)



redoRelation = redo.values.tolist()
intel = intel.values.tolist()
randomic = randomic.values.tolist()
dql = dql.values.tolist()

#print(redoRelation)
#print(redo)
#print(len(intel))





# PLOT DOS REWORKS    
a=[]
for i in range(len(redoRelation)):
    for j in range(len(intel)):
        if(redoRelation[i][1]==intel[j][0] and redoRelation[i][3]=='Supervisory+RL'):
            if(redoRelation[i][2]=='redo_A'):
                if(intel[j][1]+intel[j][3]!=0):
                    a.append((redoRelation[i][0]/(intel[j][1]+intel[j][3]), 'Rework Type 1', redoRelation[i][3], redoRelation[i][1])) 
                else:
                    a.append((0, 'Rework Type 1', redoRelation[i][3], redoRelation[i][1])) 
            else:
                if(intel[j][2]+intel[j][4]!=0):
                    a.append((redoRelation[i][0]/(intel[j][2]+intel[j][4]), 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
                else:
                    a.append((0, 'Rework Type 2', redoRelation[i][3], redoRelation[i][1])) 
        
        # tirar o espaço do 'Supervisory '
        if(redoRelation[i][3]=='Supervisory' and redoRelation[i][1]==randomic[j][0]):
            if(redoRelation[i][2]=='redo_A'):
               if(randomic[j][1]+randomic[j][3]!=0):
                     a.append((redoRelation[i][0]/(randomic[j][1]+randomic[j][3]), 'Rework Type 1', redoRelation[i][3], redoRelation[i][1])) 
               else:
                     a.append((0, 'Rework Type 1', redoRelation[i][3], redoRelation[i][1]))
            else:
                if(randomic[j][2]+randomic[j][4]!=0):
                    a.append((redoRelation[i][0]/(randomic[j][2]+randomic[j][4]), 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
                else:
                    a.append((0, 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
        if(redoRelation[i][3]=='DQL' and redoRelation[i][1]==dql[j][0]):
            if(redoRelation[i][2]=='redo_A'):
               if(dql[j][1]+dql[j][3]!=0):
                     a.append((redoRelation[i][0]/(dql[j][1]+dql[j][3]), 'Rework Type 1', redoRelation[i][3], redoRelation[i][1])) 
               else:
                     a.append((0, 'Rework Type 1', redoRelation[i][3], redoRelation[i][1]))
            else:
                if(dql[j][2]+dql[j][4]!=0):
                    a.append((redoRelation[i][0]/(dql[j][2]+dql[j][4]), 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
                else:
                    a.append((0, 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
                    
a = pd.DataFrame(a, columns=["Reworks/Cars Produced", "Event", "Method", xlabel_name])

print(a)

sns.lineplot(x=xlabel_name, y="Reworks/Cars Produced", style="Event",  hue="Method", data=a, markers=True,  err_style=None)
plt.savefig('Plots/Both/'+dataname+'/reworks-'+dataname+'.eps', dpi=300)




end = time.time()
print("Execution time: {}".format(end-start))





