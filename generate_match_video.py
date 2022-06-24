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
# for recording gym reder
import os
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
os.environ["SDL_VIDEODRIVER"] = "dummy"
from IPython import display
from tqdm import tqdm

################
#Some simple functions
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f' {episode_num+1}', fill=text_color)
    return im

def format_index(t):
    if t<10: return "000{}".format(t)
    elif t<100: return "00{}".format(t)
    elif t<1000: return "0{}".format(t)
    else: return "{}".format(t)

def generate_video(configId,seed):
    makeVideoCommand = "ffmpeg -framerate 10 -f image2 -i ./videos/img%4d.jpg -y ./videos/{0}_seed_{1}.mp4".format(configId,seed)
    process = subprocess.Popen(makeVideoCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    clearImgDataCommad = "rm ./videos/img*.jpg"
    res = subprocess.call(clearImgDataCommad, shell=True)



###############
parser = argparse.ArgumentParser(description='Create video of a Space Invaders match played by a trained SAC agent')
parser.add_argument('--config', help="Json file with all the metaparameters. See config01.json as an example.", type=str, default="config01.json",dest="config_file")
parser.add_argument('--seed', help="Seed of random number generator", type=int, default=0,dest="seed")
parser.add_argument('--game', type=str, default="BeamRider", dest="game")
parser.add_argument('--random', type=int, default=False, dest="random_mode")

args = parser.parse_args()
############

#PARAMS
print("reading parameters...")
config_file = args.config_file
seed = args.seed
game = args.game
random_mode = bool(args.random_mode)

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
# env = wrappers.AtariPreprocessing(env,grayscale_obs=True,frame_skip=0,grayscale_newaxis=True,screen_size=screen_size)

seed=random.randint(0, 9)

env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



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
state = env.reset()
state = np.transpose(state, [2,0,1])
t=0

######

frames = []        
episode_steps = 0
episode_return = 0.0
f_lives = env.unwrapped.ale.lives()
done = False
while True:
    try:
        state_cuda = torch.Tensor(state).unsqueeze(0)
        if random_mode:
          action = qnet_agent.select_action(state_cuda)
        else:
          action = qnet_agent.exploit_action(state_cuda)

        # prevent agent doesn't start at the begining
        if t==0:
          action = 2
        
        frame = env.render(mode='rgb_array')
        frames.append(_label_with_episode_number(frame, episode_num=episode_steps))
            
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
        print('move {0}/{1}'.format(t, int(1e4)))
        if t>1e4:
            env.close()
            break
        if done: 
            env.close()
            break
    except KeyboardInterrupt:
        env.close()
        print("break")
        break

print("Done. Generating Video...")
# for Recording frame
if random_mode:
  print("RANDOM MODE!")
fileName = 'SA_SAC_' + game +'.gif'
print('dir',os.path.join('./videos/', fileName))
imageio.mimwrite(os.path.join('./videos/', fileName), frames, fps=10)
#########################
print('episode Reward: ' + str(episode_return))
print('episode terminated at '+str(episode_steps)+' steps')
print("Done. Bye.")