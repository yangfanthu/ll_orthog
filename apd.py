import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import APDGaussianPolicy, QNetwork
import random
import utils
from replay_memory import ReplayMemory

class APDSAC(object):
    def __init__(self,
                num_inputs,
                action_space,
                args, 
                outdir=None, 
                alpha=0,
                buffer_max_size=int(1e6),
                suffix=None):
        self.in_dim = num_inputs
        self.out_dim = action_space.shape[0]
        self.action_space = action_space
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = alpha
        # self.writer = writer
        self.outdir = outdir
        self.batch_size = args.batch_size
        self.suffix = suffix
        self.shared_info_dim = args.shared_feature_dim
        self.num_tasks = 10
        self.bias_weight = args.bias_weight
        self.diff_weight = args.diff_weight

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.policy = APDGaussianPolicy(shared_info_dim=self.shared_info_dim).to(self.device)
        self.task_id = 0

    def select_action(self, state, task_id, evaluate=False):
        if task_id > len(self.policy.action_bias) - 1:
            return self.action_space.sample()
        else:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if evaluate is False:
                action, _, _ = self.policy.sample(state, task_id)
            else:
                _, _, action = self.policy.sample(state, task_id)
            return action.detach().cpu().numpy()[0]
    
    def add_buffer(self, state, action, next_state, reward, done, task_id):
        self.replay_buffer[task_id].push(state, action, next_state, reward, done)

    def update_parameters(self, memory_list, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory_list[self.task_id].sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, self.task_id)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch, self.task_id)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        additional_loss = self.bias_weight * self.policy.get_bias_loss() + self.diff_weight * self.policy.get_diff_loss()
        policy_loss += additional_loss 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    def save_model(self, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = os.path.join(self.outdir, "actor_{}.ckpt".format(suffix))
        if critic_path is None:
            # critic_path = os.path.join(self.outdir, "critic_{}.ckpt".format(suffix))
            critic_path = os.path.join(self.outdir, "critic.ckpt")
            critic_target_path = os.path.join(self.outdir, "critic_target.ckpt")
            # don't wanna save the critic
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        # torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.critic_target.state_dict(), critic_path)
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
    def add_task(self):
        self.alpha = self.args.alpha
        # clear the buffer for the previous task
        # if len(self.policy.linear1.tau_l) >= 1:
        #     self.replay_buffer[len(self.policy.linear1.tau_l)-1].clear()
        self.policy.add_task(self.in_dim, self.out_dim, self.args.hidden_size, action_space=self.action_space, device=self.device)
        print("Adding task {}".format(len(self.policy.linear1.tau_l) - 1))
        assert len(self.policy.linear1.tau_l) - 1 < self.num_tasks

        self.critic = QNetwork(self.in_dim, self.action_space.shape[0], self.args.hidden_size).to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)

        self.critic_target = QNetwork(self.in_dim, self.action_space.shape[0], self.args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr)
    def set_task_id(self, task_id):
        self.task_id = task_id