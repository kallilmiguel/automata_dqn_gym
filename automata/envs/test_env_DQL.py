# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:24:48 2021

@author: Lucas
"""

import gym
from DQN import Agent
#from plotting import plot_learning_curve
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500
    
    for i in range(n_games):
        count = 0
        score = 0
        done = False
        observation = env.reset()
        #print('observation: ', observation)
       # print('\n\ninitial observation: ', observation)
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            #print('\n\nmiddle observation: ', observation)
            agent.store_transition(observation, action, reward, 
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-100:])
        #print('final observation: ', observation)
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score, 
              'epsilon %.2f' % agent.epsilon)
        
        x = [i+1 for i in range(n_games)]
        filename = 'lunar_lander_2021.png'
        #plot_learning_curve(x, scores, eps_history, filename)
 
        

            