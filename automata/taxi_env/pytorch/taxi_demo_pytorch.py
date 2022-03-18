#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:09:11 2021

@author: lucas
"""

import os
import gym
from pytorch.config import Config
from pytorch.model import DQN
from pytorch.agent import QAgent

# INSTRUÇÕES: rodar um dos grupos de 6 linha (usando f9) e por último rodar o play


env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch_new.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)
agent.compile()
agent.fit()

env = gym.make("Taxi-v3").env
env.seed(100)
config = "pytorch/config_pytorch.yaml"
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





