# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:03:43 2021

@author: Lucas
"""

import gym
import random
import os
import time
import torch as T
import numpy as np
import csv
import pandas as pd
from DQN import Agent
import seaborn as sns
import matplotlib.pyplot as plt



start = time.time()
    
    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir('C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym_cont/automata/envs')

#print('cuda:0' if T.cuda.is_available() else 'cpu')

env = gym.make('automata:automata-v0')
info_dql = []
final_states_dql=[]
#print(env.action_space.n)



bad_A = 0
bad_B = 1
#good_A = 10
#good_B = 11
good_A = 12
good_B = 13
#redo_A = 12
#redo_B = 13
redo_A = 17
redo_B = 18


directory="dados_lucas/"
dataname="rework"

last_action=[bad_A, bad_B, good_A, good_B, redo_A, redo_B]

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=35,
                  eps_end=0.01, input_dims=1, lr=0.003) # mudar o input dims

cases=9
mean_reward_episodes = [0 for i in range(cases)]

for k in range(cases):
    with open('C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym_cont/automata/envs/testes_mesa/'+dataname+'/case'+str(k+1)+'.csv', newline='') as csvfile:
    #with open('C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym_cont/automata/envs/testes/'+dataname+'/case'+str(k+1)+'.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
        
    transition_names = list(map(str, data[0]))
    reward = list(map(int, data[1]))
    probabilities = list(map(float, data[2]))
    
    #env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    env.reset("SM/Renault_mesa.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    
    last_actions=last_action
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
        print("\t\tEpisode: {}, Test Case: {}, Mean Reward: {}".format(i, (k+1)*10, score))
        
        #print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
        #print('episode ', i, 'score %.2f' % score, 'epsilon %.5f' % agent.epsilon)
              #'average score %.2f' % avg_score, 
    mean_reward_episodes[k] = sum_reward_episodes/episodes
    
    #info_dql.append((mean_reward_episodes[k], 10*k, "DQL"))    
   


reward_dataname=directory+dataname+"_reward.csv"

occurrences_dql_dataname=directory+dataname+"_fsDQL.csv"
occurrences_redo_dql = directory+dataname+"_redo.csv"

#nome do eixo x do gráfico
xlabel_name="xl"

fsdql=[]
redo=[]
#last_actions.append(redo_A)
#last_actions.append(redo_B)
for i in last_actions:
    for j in range(cases):
        fsdql.append((final_states_dql.count((i,j+1)), (j+1)*10, env.mapping()[i][1]))
        #fsRdn.append((final_states_rdn.count((i,j+1)), (j+1)*10, env.mapping()[i][1]))
        if i==redo_A or i==redo_B:
            redo.append((final_states_dql.count((i,j+1)),(j+1)*10,env.mapping()[i][1],"DQL"))
            #redo.append((final_states_rdn.count((i,j+1)),(j+1)*10,env.mapping()[i][1],"Supervisory"))
     
list2dql=[]
redodql=[]
for i in range(0,9):
    list2dql.append((fsdql[i][1],fsdql[i][0], fsdql[i+9][0],fsdql[i+18][0],fsdql[i+27][0]))
    #redodql.append((fsdql[i][1],fsdql[i+36][0],fsdql[i+45][0]))
    
data = np.vstack((info_dql))
data = pd.DataFrame(data, columns=["yl",  xlabel_name, "method"])
states_dql = pd.DataFrame(list2dql,columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
#states_rdn = pd.DataFrame(list2Rdn, columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
redo = pd.DataFrame(redo, columns=["Number of Occurrences", xlabel_name, "Event", "Method"])
#redoRdn = pd.DataFrame(redoRdn, columns=[xlabel_name,"Rework Type 1","Rework Type 2"])

data.to_csv(reward_dataname)
states_dql.to_csv(occurrences_dql_dataname)
redo.to_csv(occurrences_redo_dql)
# redoRdn.to_csv(occurrences_redo_rdn)

dql = pd.read_csv(occurrences_dql_dataname)
df = pd.read_csv(reward_dataname)
redo = pd.read_csv(occurrences_redo_dql)
# redoRdn = pd.read_csv(occurrences_redo_rdn)

dql = dql.drop(["Unnamed: 0"],axis=1)
redo = redo.drop(["Unnamed: 0"], axis=1)
# redoRdn = redoRdn.drop(["Unnamed: 0"], axis=1)

sns.lineplot(data=df,  hue='method', x='xl', y='yl')
#plt.set_xlabel("Recompensa Obtida")
#plt.set_ylabel("Número de Ocorrências")
#plt.savefig('Plots/DQL/'+dataname+'/rewards-'+dataname+'.eps', dpi=300)
plt.savefig('Plots_mesa/DQL/'+dataname+'/rewards-'+dataname+'.eps', dpi=300)


plot = dql.plot.bar(x=xlabel_name, stacked=True)
plot.set_xlabel("Probabilidade")
plot.set_ylabel("Número de Ocorrências")

redoRelation = redo.values.tolist()
dql = dql.values.tolist()


a=[]
for i in range(len(redoRelation)):
    for j in range(len(dql)):
        if(redoRelation[i][1]==dql[j][0] and redoRelation[i][3]=='Supervisory+RL'):
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
        """if(redoRelation[i][3]=='Supervisory' and redoRelation[i][1]==randomic[j][0]):
            if(redoRelation[i][2]=='redo_A'):
               if(randomic[j][1]+randomic[j][3]!=0):
                     a.append((redoRelation[i][0]/(randomic[j][1]+randomic[j][3]), 'Rework Type 1', redoRelation[i][3], redoRelation[i][1])) 
               else:
                     a.append((0, 'Rework Type 1', redoRelation[i][3], redoRelation[i][1]))
            else:
                if(randomic[j][2]+randomic[j][4]!=0):
                    a.append((redoRelation[i][0]/(randomic[j][2]+randomic[j][4]), 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
                else:
                    a.append((0, 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))"""
                    
a = pd.DataFrame(a, columns=["Reworks/Cars Produced", "Event", "Method", xlabel_name])

#plt.savefig('Plots/DQL/'+dataname+'/actionstaken-'+dataname+'.eps', dpi=300)
plt.savefig('Plots_mesa/DQL/'+dataname+'/actionstaken-'+dataname+'.eps', dpi=300)


end = time.time()
print("Execution time: {}".format(end-start))





