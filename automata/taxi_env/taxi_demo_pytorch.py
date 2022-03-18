#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:09:11 2021

@author: lucas

set folder to '/home/lucas/automata_gym/automata/envs'
"""

import os
os.chdir('C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym_cont')
#os.chdir('/home/lucas/automata_gym/automata/envs')

import gym
from automata.taxi_env.pytorch.config import Config
from automata.taxi_env.pytorch.model import DQN
from automata.taxi_env.pytorch.agent import QAgent
import csv 

#os.chdir('/home/lucas/automata_gym/automata/envs')

cases=9
info_int=[]
info_rdn=[]
final_states_int=[]
final_states_rdn=[]
last_action=[0,1,10,11]
xValues=[]




# INSTRUÇÕES: rodar um dos grupos de 6 linha (usando f9) e por último rodar o play
env = gym.make("automata:automata-v0")
env.seed(100)
#env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
 
k = 0
#with open('C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym/automata/envs/testes/approve-A-90/case'+str(k+1)+'.csv', newline='') as csvfile:
with open('C:\\Users\Lucas\Google Drive\Pesquisa\TCC\automata_gym_cont\automata\envs\testes\approve-A-90/case'+str(k+1)+'.csv', newline='') as csvfile:
        #C:\Users\Lucas\Google Drive\Pesquisa\TCC\automata_gym_cont\automata\envs\testes\approve-A-90
    data = list(csv.reader(csvfile))
    #with open('testes/approve-A-90/case1.csv', newline='') as csvfile:
    #    data = list(csv.reader(csvfile))
    
    # Mapeia e armazena as recompensas e probabilidades do arquivo
    reward = list(map(int, data[1]))
    probabilities = list(map(float, data[2]))
    
    #start = time.time()
    env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    #end = time.time()
    

   
config = "C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym/automata/taxi_env/pytorch/config_pytorch.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()



'''
env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()



env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_new.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()




env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_lr_const.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()

env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_high_eps.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()

env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_low_eps.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()

env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_high_lr.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()


env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_low_lr.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()

env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_high_update.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()

env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_low_update.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()


env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_low_gamma.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()

env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_high_batch.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()

env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_low_batch.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()



#rodar essa linha por último
agent.play(sleep=0.1, max_steps=20)


'''


