import gym
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from memories import ExperienceReplay
from gym import wrappers
from gym.wrappers import AtariPreprocessing
import qnet_agentsSAC_auto
import json
import argparse

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



parser = argparse.ArgumentParser(description='Train SAC algorithm on Space Invaders')
parser.add_argument('--config', help="Json file with all the metaparameters. See config01.json as an example.", type=str, default="config01.json",dest="config_file")
parser.add_argument('--new', help="If 1 (default) the training starts from scracth, if 0 it starts from an old configuration.", type=int,default=True,dest="new_flag")

args = parser.parse_args()


##############
#PARAMS
print("reading parameters...")
config_file = args.config_file
new_flag = bool(args.new_flag)

config = json.load(open(config_file))

#Id
configId = config["configId"]

#env
screen_size = config["env_parameters"]["screen_size"]
frame_skip = config["env_parameters"]["frame_skip"]
seed_value = config["env_parameters"]["seed_value"]

#agent
gamma = config["agent_parameters"]["gamma"]
lr_Q = config["agent_parameters"]["lr_Q"]
lr_pi = config["agent_parameters"]["lr_pi"]
lr_alpha = config["agent_parameters"]["lr_alpha"]
tau = config["agent_parameters"]["tau"]
h_dim = config["agent_parameters"]["h_dim"]
h_mu_dim = config["agent_parameters"]["h_mu_dim"]
alpha = config["agent_parameters"]["alpha"]
entropy_rate = config["agent_parameters"]["entropy_rate"]

#training
n_episodes = int(config["training_parameters"]["n_episodes"])
batch_size = config["training_parameters"]["batch_size"]
t_tot_cut = config["training_parameters"]["t_tot_cut"]

##############
#SETUP
print("setting up environment and agent...")

env = gym.make('SpaceInvaders-v4')
env.spec.id = 'SpaceInvaders-v4'+"NoFrameskip"

env.seed(seed_value)
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

env = wrappers.AtariPreprocessing(env,grayscale_obs=True,frame_skip=8,grayscale_newaxis=True,screen_size=screen_size)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

QNet_Agent = qnet_agentsSAC_auto.QNet_Agent
memory = ExperienceReplay()
qnet_agent = QNet_Agent(n_states=n_states,
                        n_actions=n_actions,
                        gamma = gamma,
                        lr_Q = lr_Q,
                        lr_pi = lr_pi,
                        lr_alpha = lr_alpha,
                        tau = tau,
                        h_dim = h_dim,
                        h_mu_dim = h_mu_dim,
                        entropy_rate = entropy_rate,
                        alpha = alpha
                       ).cuda()

print(new_flag)
if not new_flag:
    print("reading old configuration from {}.json..".format(configId))
    qnet_agent.Q.load_state_dict(torch.load("./saved_models/SpaceInvaders_Q_SAC_auto_{}.model".format(configId)))
    qnet_agent.target_Q.load_state_dict(torch.load("./saved_models/SpaceInvaders_target_Q_SAC_auto_{}.model".format(configId)))
    qnet_agent.pi.load_state_dict(torch.load("./saved_models/SpaceInvaders_pi_SAC_auto_{}.model".format(configId)))


##############
#TRAIN

print("start training...")
rewards_per_episode = []; time_per_episode = []; t_total = 0


time_start = time.time()
best_episode_reward = -float("inf")

for i_episode in range(len(rewards_per_episode),n_episodes):
    env = gym.make('SpaceInvaders-v4')
    env.spec.id = 'SpaceInvaders-v4'+"NoFrameskip"
    env = wrappers.AtariPreprocessing(env,grayscale_obs=True,frame_skip=frame_skip,grayscale_newaxis=True,screen_size=screen_size)

    try:
        state = env.reset()
        state = np.transpose(state, [2,0,1])

        t=0
        rewards = []

        while True:

            t+=1
            t_total+=1

            state_cuda = torch.Tensor(state).cuda().unsqueeze(0)

            action = qnet_agent.select_action(state_cuda)
            
            new_state, reward, done, info = env.step(action)
            new_state = np.transpose(new_state, [2,0,1])
            memory.push(state, action, new_state, reward, done)
            
            if memory.__len__()>=2e4 and t%4==0: 
                batch = memory.sample(batch_size)
                qnet_agent.optimize(batch)        
            state = new_state
            rewards.append(reward)
            
            if t>=t_tot_cut: break
            if done: break


        rewards_per_episode.append(np.sum(rewards))
        best_episode_reward = max(rewards_per_episode)
        
        time_per_episode.append(t_total)
        if i_episode%1==0:

            elapsed_time = round(time.time() -  time_start)
            dbin = 1000
            s = "i_episode = {0}, t_total = {1}".format(i_episode, t_total)
            s += "\nlast avg. reward = {0}, elapsed_time = {1} sec.".format(np.mean(rewards_per_episode[-dbin:]),elapsed_time)
            s+= "\nalpha = {0}".format(qnet_agent.alpha.cpu().item())
            s+= "\nbest episode reward = {}".format(best_episode_reward)
            print("training {}".format(configId))
            print(s)
            print()
            
            if i_episode%100==0:
                plt.title(s)
                plt.plot(rewards_per_episode,alpha=0.4)
                plt.hlines(0,0,len(rewards_per_episode))            
                if len(rewards_per_episode)>dbin*2:
                    ma_rewards_per_episode = moving_average(rewards_per_episode,dbin)
                    plt.plot(ma_rewards_per_episode)   
                plt.tight_layout()
                plt.savefig("./train_figs/train_curve_{}.png".format(configId), dpi=300)
                plt.close()

        if i_episode%100==0:
            torch.save(qnet_agent.Q.state_dict(), "./saved_models/SpaceInvaders_Q_SAC_auto_{}.model".format(configId))
            torch.save(qnet_agent.target_Q.state_dict(), "./saved_models/SpaceInvaders_target_Q_SAC_auto_{}.model".format(configId))
            torch.save(qnet_agent.pi.state_dict(), "./saved_models/SpaceInvaders_pi_SAC_auto_{}.model".format(configId))
        

            
    except KeyboardInterrupt:
        print("break")
        break

