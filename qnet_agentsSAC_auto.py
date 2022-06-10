import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable
import neural_nets_SAC
import imp
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import math

imp.reload(neural_nets_SAC)
SoftQNeuralNetworkTwin = neural_nets_SAC.SoftQNeuralNetworkTwin
SoftPiNeuralNetwork = neural_nets_SAC.SoftPiNeuralNetwork

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "7"

class QNet_Agent(nn.Module):
    def __init__(self,n_states, n_actions,gamma=0.99,lr_Q=0.0001,lr_pi=0.0001,lr_alpha=0.0001,tau=0.0001, h_dim=512, h_mu_dim=512, alpha="auto", entropy_rate=0.25):
        super(QNet_Agent,self).__init__()
        
        #The soft twins##########################
        self.Q = SoftQNeuralNetworkTwin(n_states, n_actions, h_dim=h_dim).cuda()
        self.target_Q = SoftQNeuralNetworkTwin(n_states, n_actions, h_dim=h_dim).cuda()
        #copy value parameters on target
        self.target_Q.load_state_dict(self.Q.state_dict())      
      
    
        #The soft actor##########################
        self.pi = SoftPiNeuralNetwork(n_states, n_actions,h_dim=h_mu_dim).cuda()

        self.incr1 = (0.0392) / 300000
        self.epsilon_by_frame = lambda frame_idx: 0 + self.incr1 * frame_idx

        self.incr2 = (0 - 1) / 300000
        self.beta_by_frame = lambda frame_idx: 1 + self.incr2 * frame_idx


        self.Q_optimizer = optim.Adam(self.Q.parameters(),lr=lr_Q)
        self.pi_optimizer = optim.Adam(self.pi.parameters(),lr=lr_pi)

        self.loss_function = nn.MSELoss()
        self.softmax = nn.Softmax(-1)
        
        self.gamma = gamma
        if alpha!="auto":
            #fixed alpha
            self.train_alpha = False
            self.alpha = self.alpha = torch.tensor(alpha).cuda()
        else:
            #learn alpha
            self.train_alpha = True
            self.target_entropy = torch.log(torch.Tensor([n_actions])).cuda()*entropy_rate
            self.log_alpha =  torch.zeros(1, device="cuda")
            self.alpha = self.log_alpha.exp()
            self.log_alpha.requires_grad = True
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)
                                        
        self.tau = tau
        
        self.n_states = n_states
        self.n_actions = n_actions


    def select_action(self, state):

        with torch.no_grad():
            self.pi.eval()
            state = torch.FloatTensor(state).cuda()
            action = self.pi.sample(state, n_samples = 1)
            action = action.cpu().numpy()[0][0]
        return action
    
    def exploit_action(self, state):
        with torch.no_grad():
            self.pi.eval()
            state = torch.FloatTensor(state).cuda()
            action = self.pi.get_prob(state)
            action = torch.argmax(action, dim=1, keepdim=True)
        return action

    def actor_bound(self, phi_lb, phi_ub, beta=1.0, eps=None, norm=np.inf, upper=True, lower=True, phi = None, center = None):
            ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=phi_lb, x_U=phi_ub)
            x = BoundedTensor(phi, ptb)
            ilb, iub = self.pi.pi.compute_bounds(x=(x,), IBP=True, method=None)
            ilb = self.softmax(ilb.clip(-10,10))
            iub = self.softmax(iub.clip(-10,10))
            if beta > 1e-10:
                clb, cub = self.pi.pi.compute_bounds(IBP=False, method="backward")
                clb = self.softmax(clb.clip(-10,10))
                cub = self.softmax(cub.clip(-10,10))
                ub = cub * beta + iub * (1.0 - beta)
                lb = clb * beta + ilb * (1.0 - beta)
                return ub, lb
            else:
                return iub, ilb

    def optimize(self,batch, cur_step, clip_error=False, update_targets_and_policy=True, n_samples = 100, ):
        
        state, action, new_state, reward, done = batch 
        
        state = torch.Tensor(np.array(state)).cuda()#.unsqueeze(1)
        action = torch.LongTensor(np.array(action)).cuda()
        new_state = torch.Tensor(np.array(new_state)).cuda()#.unsqueeze(1)
        reward = torch.Tensor(reward).cuda()
        #reward = reward.clip(-1,1)
        done = torch.Tensor(done).cuda()
        done = done
            
        #update twins
        self.Q.train()           
        #define V(s_t+1) = Q - alpha*logprob
        Q_new_state1, Q_new_state2 = self.target_Q(new_state)
        Q_new_state = torch.min(Q_new_state1,Q_new_state2)       
        
        with torch.no_grad():
          new_state_probs = self.pi.get_prob(new_state)
          V_new = torch.sum((Q_new_state - self.alpha*torch.log(new_state_probs))*new_state_probs, dim=1)
          #define target= r + gamma*V(s_t+1)
          target_Q_value = reward + (1 - done) * self.gamma * V_new.detach()
        
        Q1,Q2 = self.Q(state)
        Q1 = torch.gather(Q1,1, action[:,None]).squeeze(1)
        Q2 = torch.gather(Q2,1, action[:,None]).squeeze(1)

        Q_loss = self.loss_function(Q1, target_Q_value) + self.loss_function(Q2, target_Q_value)
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        
        
        #if signal has arrived update policy and targets
        if update_targets_and_policy:
            ###updating the actor#####
            self.pi.train()
            
            state_probs = self.pi.get_prob(state)
            with torch.no_grad():
              Q1,Q2 = self.Q(state)
              Q = torch.min(Q1, Q2)
            

            action_reg_loss = 0
            if cur_step > 2000000:
              scaled_robust_eps = 0.0392
              robust_beta = self.beta_by_frame(cur_step - 300000)
              actor_ub, actor_lb = self.actor_bound(
                phi_lb=state - scaled_robust_eps, 
                phi_ub=state + scaled_robust_eps,
                beta=robust_beta, upper=True, lower=True, phi=state)
              phi = self.softmax(self.pi(state).clip(-10,10)) 
              action_reg_loss = torch.max(torch.pow(actor_ub - phi,2), torch.pow(actor_lb - phi,2)).mean()
              kappa = 1
              action_reg_loss = 1*torch.sqrt(action_reg_loss)
              pi_loss = torch.sum((self.alpha*torch.log(state_probs) - Q)*state_probs, dim=1).mean()
              pi_loss += action_reg_loss
            else:
              pi_loss = torch.sum((self.alpha*torch.log(state_probs) - Q)*state_probs, dim=1).mean()

            #be sure to delete these
            self.pi_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            self.pi_optimizer.step()        

            
            ##update target parameters: twins
            Q_dict = self.Q.state_dict()
            target_Q_dict = self.target_Q.state_dict()
            for k in target_Q_dict.keys():
                target_Q_dict[k] = self.tau*Q_dict[k] + (1 - self.tau)*target_Q_dict[k] 
            self.target_Q.load_state_dict(target_Q_dict)
            
            
            #train alpha 
            if self.train_alpha:
                # alpha cresce con (e_target - e)
                # Intuitively, we increse alpha when entropy is less than target
                entropy_loss = -self.log_alpha* torch.sum( (self.target_entropy + torch.log(state_probs))*state_probs, dim=1).detach()
                entropy_loss = entropy_loss.mean(dim=0).mean(dim=0)
                self.alpha_optim.zero_grad()
                entropy_loss.backward(retain_graph=True)
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp().clip(1e-4,1)
        
        return Q_loss, pi_loss, entropy_loss, action_reg_loss

