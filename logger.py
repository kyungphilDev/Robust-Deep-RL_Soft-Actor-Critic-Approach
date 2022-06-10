import time
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import glob
from collections import deque
from IPython.display import clear_output


class Logger:
    def __init__(self):
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.running_reward = 0
        self.running_alpha_loss = 0
        self.running_q_loss = 0
        self.running_policy_loss = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log(self, *args):

        episode, episode_reward, alpha_loss, q_loss, policy_loss, step, eval_reward, action_reg_loss = args

        self.max_episode_reward = max(self.max_episode_reward, episode_reward)

        if self.running_reward == 0:
            self.running_reward = episode_reward
            self.running_alpha_loss = alpha_loss
            self.running_q_loss = q_loss
        else:
            self.running_alpha_loss = 0.99 * self.running_alpha_loss + 0.01 * alpha_loss
            self.running_q_loss = 0.99 * self.running_q_loss + 0.01 * q_loss
            self.running_policy_loss = 0.99 * self.running_policy_loss + 0.01 * policy_loss
            self.running_reward = 0.99 * self.running_reward + 0.01 * episode_reward

        self.last_10_ep_rewards.append(int(episode_reward))
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            last_10_ep_rewards = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')
        else:
            last_10_ep_rewards = 0  # It is not correct but does not matter.


        with SummaryWriter("Logs/" + self.log_dir) as writer:
            if eval_reward != -1:
              writer.add_scalar("Stats/Evaulation reward", eval_reward, step)  
            writer.add_scalar("Stats/Episode running reward", self.running_reward, step)
            writer.add_scalar("Stats/Max episode reward", self.max_episode_reward, step)
            writer.add_scalar("Stats/Moving average reward of the last 10 episodes", last_10_ep_rewards, step)
            writer.add_scalar("Loss/Alpha Loss", alpha_loss, step)
            writer.add_scalar("Loss/Q-Loss", q_loss, step)
            writer.add_scalar("Loss/Policy Loss", policy_loss, step)
            writer.add_scalar("Loss/Reg Loss", action_reg_loss, step)
