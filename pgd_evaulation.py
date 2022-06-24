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
from tqdm import tqdm
# PGD Attack Modules
class Attacker(ABC):
    def __init__(self, model, config):
        """
        ## initialization ##
        :param model: Network to attack
        :param config : configuration to init the attack
        """
        self.config = config
        self.model = model
        self.clamp = (0,1)
        self.device = torch.device("cuda")        
            
    def _random_init(self, x):
#         x = torch.ByteTensor(
#             x[None, ...]).to(self.device).float() / 255.
        x = x + (torch.rand(x.size(), dtype=torch.float, device=self.device) - 0.5) * 2 * self.config['eps']
        x = torch.clamp(x,*self.clamp)
        return x

    def __call__(self, x,y):
        x_adv = self.forward(x,y)
        return x_adv
    
class PGD(Attacker):
    def __init__(self, model, config, target=None):
        super(PGD, self).__init__(model, config)
        self.target = target
        self.device = torch.device("cuda")

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        x /= 255.0
        y = torch.tensor([y]).cuda()
        x_adv = x
        if self.config['random_init'] :
            x_adv = self._random_init(x_adv)
        for _ in range(self.config['attack_steps']):
            x_adv.requires_grad = True
            self.model.zero_grad()
            # action에 대한 q value들
            logits = self.model(x_adv) #f(T((x))
            # Untargeted attacks - gradient ascent
            loss = F.cross_entropy(logits[0], y,  reduction="sum")
            loss.backward()                      
            grad = x_adv.grad.detach()
            grad = grad.sign()
            x_adv = x_adv + self.config['attack_lr'] * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.config['eps'], max=self.config['eps'])
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)
        return x_adv

parser = argparse.ArgumentParser(description='Create video of a Space Invaders match played by a trained SAC agent')
parser.add_argument('--config', help="Json file with all the metaparameters. See config01.json as an example.", type=str, default="config01.json",dest="config_file")
parser.add_argument('--seed', help="Seed of random number generator", type=int, default=0,dest="seed")
parser.add_argument('--game', type=str, default="BeamRider", dest="game")
parser.add_argument('--iter', type=int, default=10, dest="iter")
parser.add_argument('--steps', type=int, default=10, dest="attack_steps")

args = parser.parse_args()


############

#PARAMS
print("reading parameters...")
config_file = args.config_file
seed = args.seed
game = args.game
iter = int(args.iter)
attack_steps = int(args.attack_steps)
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

gameID = game + "-v4"
env = gym.make(gameID)
env.spec.id = gameID+"NoFrameskip"
env = wrappers.AtariPreprocessing(env,grayscale_obs=True,grayscale_newaxis=True,screen_size=screen_size)


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
qnet_agent.Q.load_state_dict(torch.load("./saved_models/{}_Q_SAC_auto_{}.model".format(game, configId)))
qnet_agent.target_Q.load_state_dict(torch.load("./saved_models/{}_target_Q_SAC_auto_{}.model".format(game, configId)))
qnet_agent.pi.load_state_dict(torch.load("./saved_models/{}_pi_SAC_auto_{}.model".format(game, configId)))

##################
attack_config = {
  'eps' : 8.0/255.0,
  'attack_steps': attack_steps,
  'attack_lr': 1 / 255.0,
  'random_init': False
}

cur_model = qnet_agent.Q
attacker = PGD(cur_model, attack_config)

rewards = []
max_reward = -1
min_reward = 2**31-1



for i in tqdm(range(iter)):
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
          true_action = qnet_agent.exploit_action(state_cuda)

          state_cuda = torch.Tensor(state).cuda().unsqueeze(0)

          x_adv = attacker(state_cuda, true_action)
          state_max = 255.0
          x_adv = torch.clamp(x_adv*255, max=state_max).cpu()
          action = qnet_agent.exploit_action(x_adv)

          if t==0:
            action = 2
          
          if t % 30 == 0:
              print('PGD:', str(t), 'lifes:', f_lives ,'Reward:', episode_return)
    
          new_state, reward, done, info = env.step(action)
          ####
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