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
import json
import qnet_agentsSAC_auto
import subprocess
import argparse

################
#Some simple functions

def format_index(t):
    if t<10: return "000{}".format(t)
    elif t<100: return "00{}".format(t)
    elif t<1000: return "0{}".format(t)
    else: return "{}".format(t)

###############

parser = argparse.ArgumentParser(description='Plays 100 episodes and returns the highest match reward')
parser.add_argument('--config', help="Json file with all the metaparameters. See config01.json as an example.", type=str, default="config01.json",dest="config_file")
parser.add_argument('--seed', help="Seed of random number generator", type=str, default=0,dest="seed")
parser.add_argument('--episodes', help="Number of Matches to Play", type=str, default=100,dest="n_episodes")

args = parser.parse_args()


############

#PARAMS
print("reading parameters...")
config_file = args.config_file
seed = args.seed

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
n_episodes = int(args.n_episodes)
batch_size = config["training_parameters"]["batch_size"]
t_tot_cut = config["training_parameters"]["t_tot_cut"]

###########
#SETUP
print("setting up environment and agent...")

env = gym.make('SpaceInvaders-v4')
env.spec.id = 'SpaceInvaders-v4'+"NoFrameskip"

env = wrappers.AtariPreprocessing(env,grayscale_obs=True,frame_skip=frame_skip,grayscale_newaxis=True,screen_size=screen_size)

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

QNet_Agent = qnet_agentsSAC_auto.QNet_Agent
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
qnet_agent.Q.load_state_dict(torch.load("./saved_models/SpaceInvaders_Q_SAC_auto_{}.model".format(configId)))
qnet_agent.target_Q.load_state_dict(torch.load("./saved_models/SpaceInvaders_target_Q_SAC_auto_{}.model".format(configId)))
qnet_agent.pi.load_state_dict(torch.load("./saved_models/SpaceInvaders_pi_SAC_auto_{}.model".format(configId)))

##################

print("playng {0} episodes with {1} and seed {0}..".format(n_episodes,configId, seed))

env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
rewards_per_episode = []; 
for n in range(n_episodes):
    
    print("starting episode {}".format(n))
    
    env = gym.make('SpaceInvaders-v4')
    env.spec.id = 'SpaceInvaders-v4'+"NoFrameskip"
    env = wrappers.AtariPreprocessing(env,grayscale_obs=True,frame_skip=frame_skip,grayscale_newaxis=True,screen_size=screen_size)

    state = env.reset()
    state = np.transpose(state, [2,0,1])
    t=0
    rewards = []
    while True:
        try:
            state_cuda = torch.Tensor(state).cuda().unsqueeze(0)
            action = qnet_agent.select_action(state_cuda)
            new_state, reward, done, info = env.step(action) 
            new_state = np.transpose(new_state, [2,0,1])
            state = new_state

            time.sleep(0.00001)
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            ax.imshow(np.transpose(state, [1,2,0]))        
            plt.savefig("./videos/img{}.jpg".format(format_index(t)))
            plt.close()
            rewards.append(reward)
            t+=1
            if t>t_tot_cut:
                env.close()
                break
            if done: 
                env.close()
                break
        except KeyboardInterrupt:
            env.close()
            print("break")
            break
    rewards_per_episode.append(sum(rewards))
    print('total reward: {}'.format(rewards_per_episode[-1]))
    print()

print("Best of {0} episodes = {1}".format(n_episodes, max(rewards_per_episode)))
print("Done. Bye.")