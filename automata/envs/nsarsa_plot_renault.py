#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 18:22:16 2021

@author: lucas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:49:07 2020

@author: kallil & lucas
"""
#%%
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
info_int=[]
info_rdn=[]
final_states_int=[]
final_states_rdn=[]
#last_action=[0,1,10,11]

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
dataname="approve-A-90"
last_action=[bad_A, bad_B, good_A, good_B]
last_actions=[bad_A, bad_B, good_A, good_B, redo_A, redo_B]
print(last_actions)

xValues=[]

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
        
        info_int.append((total_reward, xValues[k], "Supervisory+RL"))
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
        
        info_rdn.append((total_reward, xValues[k], "Supervisory"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))

# Alterar dataname para salvar diferentes bases de dados
#directory="dados/"
#dataname="approve-A-10"
reward_dataname=directory+dataname+"_reward.csv"
occurrences_int_dataname=directory+dataname+"_fsInt.csv"
occurrences_rnd_dataname=directory+dataname+"_fsRnd.csv"
occurrences_redo_int = directory+dataname+"_redo.csv"

print(final_states_int)
print(final_states_rdn)

print(last_actions)

#nome do eixo x do gráfico
xlabel_name="xl"

fsInt=[]
fsRdn=[]
redo=[]
#last_actions.append(12)
#last_actions.append(13)
for i in last_actions:
    for j in range(cases):
        fsInt.append((final_states_int.count((i,j+1)), xValues[j], env.mapping()[i][1]))
        fsRdn.append((final_states_rdn.count((i,j+1)), xValues[j], env.mapping()[i][1]))
        if i==redo_A or i==redo_B:
            #print((final_states_int.count((i,j+1)), xValues[j], env.mapping()[i][1],"Supervisory+RL"))
            redo.append((final_states_int.count((i,j+1)),xValues[j],env.mapping()[i][1],"Supervisory+RL"))
            redo.append((final_states_rdn.count((i,j+1)),xValues[j],env.mapping()[i][1],"Supervisory"))
            #print(redo)


print(redo)

list2Int=[]
redoInt=[]
redoRdn=[]

print(fsInt)


for i in range(0,9):
    list2Int.append((fsInt[i][1],fsInt[i][0], fsInt[i+9][0],fsInt[i+18][0],fsInt[i+27][0]))
    redoInt.append((fsInt[i][1],fsInt[i+36][0],fsInt[i+45][0]))

list2Rdn=[]
for i in range(0,9):
    list2Rdn.append((fsRdn[i][1],fsRdn[i][0], fsRdn[i+9][0],fsRdn[i+18][0],fsRdn[i+27][0]))
    redoRdn.append((fsRdn[i][1],fsRdn[i+36][0],fsRdn[i+45][0]))




data = np.vstack((info_int, info_rdn))
data = pd.DataFrame(data, columns=["yl",  xlabel_name, "method"])
states_int = pd.DataFrame(list2Int,columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
states_rdn = pd.DataFrame(list2Rdn, columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
redo = pd.DataFrame(redo, columns=["Number of Occurrences", xlabel_name, "Event", "Method"])
#redoRdn = pd.DataFrame(redoRdn, columns=[xlabel_name,"Rework Type 1","Rework Type 2"])
#print(redo)

data.to_csv(reward_dataname)
states_int.to_csv(occurrences_int_dataname)
states_rdn.to_csv(occurrences_rnd_dataname)
redo.to_csv(occurrences_redo_int)
#redoRdn.to_csv(occurrences_redo_rdn)
#print(redo)


intel = pd.read_csv(occurrences_int_dataname)
randomic = pd.read_csv(occurrences_rnd_dataname)
df = pd.read_csv(reward_dataname)
redo = pd.read_csv(occurrences_redo_int)
#redoRdn = pd.read_csv(occurrences_redo_rdn)
#print(redo)


intel = intel.drop(["Unnamed: 0"],axis=1)
randomic = randomic.drop(["Unnamed: 0"], axis=1)
redo = redo.drop(["Unnamed: 0"], axis=1)
#redoRdn = redoRdn.drop(["Unnamed: 0"], axis=1)
#print(redo)


#plot = sns.lineplot(data=df,  hue='method', x='xl', y='yl')
sns.lineplot(data=df,  hue='method', x='xl', y='yl', ci=1)
plt.savefig('Plots/Sarsa/'+dataname+'/rewards-'+dataname+'.eps', dpi=300)
#plt.savefig('Plots_mesa/Sarsa/'+dataname+'/rewards-'+dataname+'.eps', dpi=300)


plot = intel.plot.bar(x=xlabel_name, stacked=True)
plot.set_xlabel(xlabel_name)
plot.set_ylabel("Number of Occurrences")
plt.savefig('Plots/Sarsa/'+dataname+'/sarsa-actionstaken-'+dataname+'.eps', dpi=300)
#plt.savefig('Plots_mesa/Sarsa/'+dataname+'/sarsa-actionstaken-'+dataname+'.eps', dpi=300)


plot = randomic.plot.bar(x=xlabel_name, stacked=True)
plot.set_xlabel(xlabel_name)
plot.set_ylabel("Number of Occurrences")
plt.savefig('Plots/Sarsa/'+dataname+'/SCT-actionstaken-'+dataname+'.eps', dpi=300)
#plt.savefig('Plots_mesa/Sarsa/'+dataname+'/SCT-actionstaken-'+dataname+'.eps', dpi=300)




redoRelation = redo.values.tolist()
intel = intel.values.tolist()
randomic = randomic.values.tolist()

print(redoRelation)

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
                    
a = pd.DataFrame(a, columns=["Reworks/Cars Produced", "Event", "Method", xlabel_name])


sns.lineplot(x=xlabel_name, y="Reworks/Cars Produced", style="Event",  hue="Method", data=a, markers=True,  err_style=None)
plt.savefig('Plots/Sarsa/'+dataname+'/rewroks-'+dataname+'.eps', dpi=300)


#plt.savefig('Plots/Sarsa/'+dataname+'/actionstaken-'+dataname+'.eps', dpi=300)

#sns.lineplot(x=xlabel_name, y="Reworks/Cars Produced", style="Method", hue="Event", data=a, markers=True)


# plt = redoInt.plot.bar(x=xlabel_name, stacked=True)
# plt.set_xlabel(xlabel_name)
# plt.set_ylabel("Number of Occurrences")

# plt = redoRdn.plot.line(x=xlabel_name, stacked=True)
# plt.set_xlabel(xlabel_name)
# plt.set_ylabel("Number of Occurrences")



end = time.time()
print("Execution time: {}".format(end-start))





