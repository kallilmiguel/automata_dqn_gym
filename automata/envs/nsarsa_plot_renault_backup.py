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

import gym
import random
import numpy as np
import time
#from policy import CustomEpsGreedyQPolicy
import matplotlib.pyplot as plt
import csv 
import pandas as pd
import seaborn as sns


##########################################################################################
#                                                                                        #
#                                                                                        #
#                       Q-LEARNING (SARSA)                                               #
#                                                                                        #
#                                                                                        #
##########################################################################################


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
last_action=[0,1,10,11]
xValues=[]


# Abre o arquivo das recompensas e armazena em data
k = 0

for k in range(cases):
    with open('testes/approve-A-90/case'+str(k+1)+'.csv', newline='') as csvfile:
            data = list(csv.reader(csvfile))
    #with open('testes/approve-A-90/case1.csv', newline='') as csvfile:
    #    data = list(csv.reader(csvfile))
    
    # Mapeia e armazena as recompensas e probabilidades do arquivo
    reward = list(map(int, data[1]))
    probabilities = list(map(float, data[2]))
    
    start = time.time()
    env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    end = time.time()
    
    print(end-start)
    
    last_actions=last_action
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    
    q_table = np.zeros([num_states, num_actions], dtype = np.float32)
    
    xValues.append(100*env.probs[0])
    
    lr = 0.1
    gamma = 0.9
    epsilon = 0.5
    
    bad=[1011]
    
    
    # ----------------------- Treinamento ----------------------
    episodes = 300
    n=10
    start = time.time()
    for i in range(episodes):
        steps=0
        print("----- Episode: {}/{}".format(i,episodes))
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
    
    end = time.time()
    print("Execution time: {}".format(end-start))
    
    # ---------------------- Teste --------------------------
    
    # Testando decisões inteligentes
    epsilon=0
    episodes = 50
    
    print("\tTeste Inteligente")
    for i in range(episodes):
        state = env.reset()
        total_reward_int=0
        
        #sum_total_reward_int=0
        #env.render()
        
        done = False
        while not done:
            
            action = epsilon_greedy(env, epsilon, q_table, state)
            if(action in last_actions or action ==12 or action ==13):
                #final_states_int.append((action,1))
                final_states_int.append((action,k+1))
            
    #            if(state in l and (action!=20 and action !=22)):
     #              print("State:{}/Action:{}".format(state,action))
                
           
            next_state, reward, done, info = env.step(action)
            total_reward_int+=reward
            
            state = next_state
        
        #info_int.append((total_reward_int, xValues[0], "Supervisory+RL"))
        info_int.append((total_reward_int, xValues[k], "Supervisory+RL"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward_int))
        
        #sum_total_reward_int += total_reward_int
    
        
    # Testando decisões aleatórias
    epsilon=1
    episodes = 50
    rewards_rdn=[]
    print("\tTeste Aleatório")
    for i in range(episodes):
        state = env.reset()
        total_reward_rnd=0
        
        #sum_total_reward_rnd=0
        #env.render()
        
        done = False
        cars=0
        count=0
        
        while not done:
    
            action = epsilon_greedy(env, epsilon, q_table, state)
            
            if(action in last_actions or action==12 or action==13):
                #final_states_rdn.append((action,1))
                final_states_rdn.append((action,k+1))
    
            
            next_state, reward, done, info = env.step(action)
            total_reward_rnd+=reward
            
            state = next_state
        
        info_rdn.append((total_reward_rnd, xValues[k], "Supervisory"))
        #info_rdn.append((total_reward, xValues[0], "Supervisory"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward_rnd))
        
        #sum_total_reward_rnd += total_reward_rnd
    
    
    # -------------- Recompensas Médias ----------------------
    #avg_int = sum_total_reward_int/episodes
    #avg_rnd = sum_total_reward_rnd/episodes
    #print("\nRecompensa média inteligente: {}\nRecompensa média aleatória: {}".format(avg_int, avg_rnd))

directory="dados_lucas/"
dataname="approve-A-90"
reward_dataname=directory+dataname+"_reward.csv"
occurrences_int_dataname=directory+dataname+"_fsInt.csv"
occurrences_rnd_dataname=directory+dataname+"_fsRnd.csv"
occurrences_redo_int = directory+dataname+"_redo.csv"


#nome do eixo x do gráfico
xlabel_name="xl"
ylabel_name="yl"

fsInt=[]
fsRdn=[]
redo=[]
last_actions.append(12)
last_actions.append(13)
#print(last_actions)

for i in last_actions:
    for j in range(cases):
        #print("-------------------\n final_states")
        #print(final_states_int.count((i,j+1)))
        #print("-------------------\n xValues")
        #print(xValues[j])
        #print("-------------------\n env.mapping")
        #print(env.mapping()[i][1])
        fsInt.append( ( final_states_int.count((i,j+1)), xValues[j], env.mapping()[i][1] ) )
        fsRdn.append((final_states_rdn.count((i,j+1)), xValues[j], env.mapping()[i][1]))
        if i==12 or i==13:
            redo.append((final_states_int.count((i,j+1)),xValues[j],env.mapping()[i][1],"Supervisory+RL"))
            redo.append((final_states_rdn.count((i,j+1)),xValues[j],env.mapping()[i][1],"Supervisory"))

   
list2Int=[]
redoInt=[]
redoRdn=[]
#for i in range(0,9):
for i in range(cases):
    list2Int.append((fsInt[i][1],fsInt[i][0], fsInt[i+cases][0],fsInt[i+2*cases][0],fsInt[i+3*cases][0]))
    #list2Int.append((fsInt[i][1],fsInt[i][0], fsInt[i+9][0],fsInt[i+18][0],fsInt[i+27][0]))
    #redoInt.append((fsInt[i][1],fsInt[i+36][0],fsInt[i+45][0]))

list2Rdn=[]
for i in range(0,9):
#for i in range(cases):
    list2Rdn.append((fsRdn[i][1],fsRdn[i][0], fsRdn[i+cases][0],fsRdn[i+2*cases][0],fsRdn[i+3*cases][0]))
    #list2Rdn.append((fsRdn[i][1],fsRdn[i][0], fsRdn[i+9][0],fsRdn[i+18][0],fsRdn[i+27][0]))
    #redoRdn.append((fsRdn[i][1],fsRdn[i+36][0],fsRdn[i+45][0]))


data = np.vstack((info_int, info_rdn))
#data = pd.DataFrame(data, columns=["yl",  xlabel_name, "method"])
data = pd.DataFrame(data, columns=[ylabel_name,  xlabel_name, 'method'])
states_int = pd.DataFrame(list2Int,columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
states_rdn = pd.DataFrame(list2Rdn, columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
redo = pd.DataFrame(redo, columns=["Number of Occurrences", xlabel_name, "Event", "Method"])
#redoRdn = pd.DataFrame(redoRdn, columns=[xlabel_name,"Rework Type 1","Rework Type 2"])

data.to_csv(reward_dataname)
states_int.to_csv(occurrences_int_dataname)
states_rdn.to_csv(occurrences_rnd_dataname)
redo.to_csv(occurrences_redo_int)
# redoRdn.to_csv(occurrences_redo_rdn)


intel = pd.read_csv(occurrences_int_dataname)
randomic = pd.read_csv(occurrences_rnd_dataname)
df = pd.read_csv(reward_dataname)
redo = pd.read_csv(occurrences_redo_int)
# redoRdn = pd.read_csv(occurrences_redo_rdn)


intel = intel.drop(["Unnamed: 0"],axis=1)
randomic = randomic.drop(["Unnamed: 0"], axis=1)
redo = redo.drop(["Unnamed: 0"], axis=1)
# redoRdn = redoRdn.drop(["Unnamed: 0"], axis=1)


#print(reward_dataname)
#print(df)
#plot = sns.lineplot(data=df, x=xlabel_name, y="mean reward", hue="method")
plot = sns.lineplot(data=df,  hue='method', x='xl', y='yl')


plt = intel.plot.bar(x=xlabel_name, stacked=True)
plt.set_xlabel(xlabel_name)
plt.set_ylabel("Number of Occurrences")

plt = randomic.plot.bar(x=xlabel_name, stacked=True)
plt.set_xlabel(xlabel_name)
plt.set_ylabel("Number of Occurrences")

redoRelation = redo.values.tolist()
intel = intel.values.tolist()
randomic = randomic.values.tolist()


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

sns.lineplot(x=xlabel_name, y="Reworks/Cars Produced", style="Method", hue="Event", data=a, markers=True)



##########################################################################################
#                                                                                        #
#                                                                                        #
#                       DEEP Q-LEARNING (TD3)                                            #
#                                                                                        #
#                                                                                        #
##########################################################################################
"""
import os
import pybullet_envs # apagaar depois
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque

##################################################################################
class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

##################################################################################
class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x

##################################################################################
class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1

##################################################################################
# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

# Building the whole Training Process into a class
class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


##################################################################################
# POLICY EVALUATION
def evaluate_policy(policy, eval_episodes=10):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

##################################################################################
# PARAMETER SETTING
env_name = "automata-v0"
#env_name = "AntBulletEnv-v0" # Name of a environment (set it to any Continous environment you want)
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 1e4#5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

##################################################################################
# FILE NAME -- REMOVER DEPOIS?
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

##################################################################################
# CREATES THE DIRECTORY FOR SAVING THE FILES
if not os.path.exists("./results_TD3"):
  os.makedirs("./results_TD3")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")
  
  
##################################################################################
# CREATES AND SEEDS THE ENVIRONMENT (AQUI TÁ DANDO RUIM)
#env = gym.make(env_name)
env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim =100# env.observation_space.shape[0]
action_dim = 30#env.action_space.shape[0]
#max_action = float(env.action_space.high[0])
max_action = float(env.space.n)
print(env.action_space)

##################################################################################
# CREATES THE POLICY
policy = TD3(state_dim, action_dim, max_action)


##################################################################################
# CREATES THE EXPERIENCE REPLAY MEMORY
replay_buffer = ReplayBuffer()

##################################################################################
# DEFINING A LIST FOR EVALUATING RESULTS OVER THE EPISODES
evaluations = [evaluate_policy(policy)]
  
##################################################################################
# CREATES DIRECTORY FOR STORING THE VIDEOS
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = env._max_episode_steps
save_env_vid = False
if save_env_vid:
  env = wrappers.Monitor(env, monitor_dir, force = True)
  env.reset()

##################################################################################
# INITIALIZES THE VARIABLES
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

##################################################################################
# TRAINING
# We start the main loop over 500,000 timesteps
while total_timesteps < max_timesteps:
  
  # If the episode is done
  if done:

    # If we are not at the very beginning, we start the training process of the model
    if total_timesteps != 0:
      print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
      policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

    # We evaluate the episode and we save the policy
    if timesteps_since_eval >= eval_freq:
      timesteps_since_eval %= eval_freq
      evaluations.append(evaluate_policy(policy))
      policy.save(file_name, directory="./pytorch_models")
      np.save("./results_TD3/%s" % (file_name), evaluations)
    
    # When the training step is done, we reset the state of the environment
    obs = env.reset()
    
    # Set the Done to False
    done = False
    
    # Set rewards and episode timesteps to zero
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1
  
  # Before 10000 timesteps, we play random actions
  if total_timesteps < start_timesteps:
    action = env.action_space.sample()
  else: # After 10000 timesteps, we switch to the model
    action = policy.select_action(np.array(obs))
    # If the explore_noise parameter is not 0, we add noise to the action and we clip it
    if expl_noise != 0:
      action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
  
  # The agent performs the action in the environment, then reaches the next state and receives the reward
  new_obs, reward, done, _ = env.step(action)
  
  # We check if the episode is done
  done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
  
  # We increase the total reward
  episode_reward += reward
  
  # We store the new transition into the Experience Replay memory (ReplayBuffer)
  replay_buffer.add((obs, new_obs, action, reward, done_bool))

  # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
  obs = new_obs
  episode_timesteps += 1
  total_timesteps += 1
  timesteps_since_eval += 1

# We add the last policy evaluation to our list of evaluations and we save our model
evaluations.append(evaluate_policy(policy))
if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results_TD3/%s" % (file_name), evaluations)

##################################################################################
# INFERENCE
class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x)) 
    return x

class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

def evaluate_policy(policy, eval_episodes=10):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

env_name = "AntBulletEnv-v0"
seed = 0

file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

eval_episodes = 10
save_env_vid = True
env = gym.make(env_name)
max_episode_steps = env._max_episode_steps
if save_env_vid:
  env = wrappers.Monitor(env, monitor_dir, force = True)
  env.reset()
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(state_dim, action_dim, max_action)
policy.load(file_name, './pytorch_models/')
_ = evaluate_policy(policy, eval_episodes=eval_episodes)

## Se der erro no monitor_dir, executar o trecho da linha 270







"""












