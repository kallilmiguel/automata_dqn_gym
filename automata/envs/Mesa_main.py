#%%
import gym
import os
import time
import torch 
import torch.optim as optim
import csv
from lib import agent_buffer
from lib import dqn_model
from operator import itemgetter
from tensorboardX import SummaryWriter
import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--gamma', type=float, default=0.99, help="Value of gamma for training")
    parser.add_argument('--n_steps', type=int, default=4, help="Number of steps to consider for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size (use higher if you have enough VRAM)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate of training")
    parser.add_argument('--dataname', type=str, default="approve-A-90", help="Case to use in the examples")
    parser.add_argument('--sync_target_frames', type=int, default=1e3, help="Number of steps for synchronization between both networks")
    parser.add_argument('--episodes', type=int, default=500, help="Learning rate of training")
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
    dataname = args.dataname

    actions = {}
    with open('./testes_mesa/'+dataname+f'/case{CASE}.csv', newline='') as file:
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

    net = dqn_model.DeepQNetwork2(lr=LEARNING_RATE, input_dims=num_states, emb_dims=4, fc1_dims=128, fc2_dims=128, n_actions=len(actions), device=device)
    tgt_net = dqn_model.DeepQNetwork2(lr=LEARNING_RATE, input_dims=num_states, emb_dims=4, fc1_dims=128, fc2_dims=128, n_actions=len(actions), device=device)
    tgt_net.load_state_dict(net.state_dict())

    writer = SummaryWriter(comment=f"/DQN:dataname={dataname}.{CASE}_n={N_STEPS}_lr={LEARNING_RATE}_gamma={GAMMA}_epsDecay={EPSILON_DECAY_LAST_STEP}")


    buffer = agent_buffer.ExperienceBuffer(REPLAY_SIZE, gamma = GAMMA, n_steps=N_STEPS)
    agent = agent_buffer.Agent(env, buffer, reward_max=max(reward))

    optimizer = optim.RMSprop(net.parameters(), lr =LEARNING_RATE)

    print("Starting training")
    episode_idx = 0
    step_idx = 0
    episodes = EPISODES
    ts = time.time()
    best_total_reward = None
    epsilon = 1
    for i in range(episodes):
        total_reward = 0
        print(f"Episode: {i}")
        observation = agent._reset()
        while True:
            step_idx+=1
            reward, done = agent.play_step(net, epsilon)
            total_reward += reward
                
            if done:
                print(f"Episode ended with reward {total_reward}")
                writer.add_scalar("total_reward_train", total_reward, i)
                if best_total_reward is None or best_total_reward < total_reward:
                    if best_total_reward is not None:
                        print("Best reward updated %.3f -> %.3f" %(best_total_reward, total_reward))
                    best_total_reward = total_reward
                break
            
            if len(buffer) < REPLAY_START_SIZE:
                continue
            
            if step_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
                
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = agent.calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()
            writer.add_scalar("loss_train", loss_t, step_idx)
    writer.close()
    
    
    print("NOW STARTING TEST PHASE")
    episode_idx = 0
    step_idx = 0
    episodes = EPISODES
    ts = time.time()
    best_total_reward = None
    epsilon = 0
    for i in range(episodes):
        total_reward = 0
        print(f"Episode: {i}")
        observation = agent._reset()
        while True:
            step_idx+=1
            reward, done = agent.play_step(net, epsilon)
            total_reward += reward
                
            if done:
                print(f"Episode ended with reward {total_reward}")
                writer.add_scalar("total_reward_test", total_reward, i)
                if best_total_reward is None or best_total_reward < total_reward:
                    if best_total_reward is not None:
                        print("Best reward updated %.3f -> %.3f" %(best_total_reward, total_reward))
                    best_total_reward = total_reward
                break
            
                
            
            
            
            
    
# %%
