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

from abc import ABC
import torch
from torch import autograd
from torch.nn import functional as F
parser = argparse.ArgumentParser(description='Create video of a Space Invaders match played by a trained SAC agent')
parser.add_argument('--config', help="Json file with all the metaparameters. See config01.json as an example.", type=str, default="config01.json",dest="config_file")
parser.add_argument('--seed', help="Seed of random number generator", type=int, default=0,dest="seed")

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
n_episodes = int(config["training_parameters"]["n_episodes"])
batch_size = config["training_parameters"]["batch_size"]
t_tot_cut = config["training_parameters"]["t_tot_cut"]

###########
#SETUP
print("setting up environment and agent...")
print("playng the match with {0} and seed {1}..".format(configId, seed))

env = gym.make('Qbert-v4')
env.spec.id = 'Qbert-v4'+"NoFrameskip"
env = wrappers.AtariPreprocessing(env,grayscale_obs=True,grayscale_newaxis=True,screen_size=screen_size)
# env = wrappers.AtariPreprocessing(env,grayscale_obs=True,frame_skip=0,grayscale_newaxis=True,screen_size=screen_size)




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
qnet_agent.Q.load_state_dict(torch.load("./saved_models (Qbert_2M+300K)/Qbert_Q_SAC_auto_{}.model".format(configId)))
qnet_agent.target_Q.load_state_dict(torch.load("./saved_models (Qbert_2M+300K)/Qbert_target_Q_SAC_auto_{}.model".format(configId)))
qnet_agent.pi.load_state_dict(torch.load("./saved_models (Qbert_2M+300K)/Qbert_pi_SAC_auto_{}.model".format(configId)))

##################




######
# for recording gym reder
import os
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
os.environ["SDL_VIDEODRIVER"] = "dummy"
from IPython import display

from tqdm import tqdm
# for recording gym reder
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f': {episode_num+1}', fill=text_color)
    return im


rewards = []
max_reward = -1
min_reward = 2**31-1



for i in tqdm(range(10)):
  state = env.reset()
  state = np.transpose(state, [2,0,1])
  t=0
  
  seed=random.randint(0, 9)
  env.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)

  episode_steps = 0
  episode_return = 0.0
  f_lives = env.unwrapped.ale.lives()
  done = False

  while True:
      try:
          state_cuda = torch.Tensor(state).unsqueeze(0)
          # if t % 3 == 0:
          #   action = qnet_agent.select_action(state_cuda)
          # else:
          #   action = qnet_agent.exploit_action(state_cuda)
          action = qnet_agent.exploit_action(state_cuda)
          if t==0:
            action = 2
    
          new_state, reward, done, info = env.step(action)

          lives = env.unwrapped.ale.lives()
          if lives < f_lives:
            f_lives = lives
            env.step(2)
          ### 
          episode_return += reward
          new_state = np.transpose(new_state, [2,0,1])
          state = new_state
          t+=1
          if done or t>1e4: 
              if episode_return > max_reward:
                max_reward = episode_return
              if episode_return < min_reward:
                  min_reward = episode_return
              rewards.append(episode_return)
              env.close()
              break
      except KeyboardInterrupt:
          env.close()
          print("break")
          break

  #########################
  print('episode Reward: ' + str(episode_return))

print('mean:', np.mean(rewards), 'std:', np.std(rewards) , 'max:', max_reward, 'min:', min_reward)
print("Done. Bye.")