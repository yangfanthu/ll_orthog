import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, EWCGaussianPolicy, QNetwork, EWCDeterministicPolicy
import random
import copy

class EWCSAC(object):
    def __init__(self, num_inputs, action_space, num_tasks, args, outdir=None):
        
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.num_tasks = num_tasks
        self.outdir = outdir

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critics = nn.ModuleList()
        self.critic_optims = []
        self.critic_targets = nn.ModuleList()
        self.task_id = 0
        self.previous_policy = []
        for i in range(num_tasks):
            self.critics.append(QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device))
            self.critic_optims.append(Adam(self.critics[i].parameters(), lr=args.lr))
            self.critic_targets.append(QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device))
            hard_update(self.critics[i], self.critic_targets[i])
        
        # self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        # self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        # hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = EWCGaussianPolicy(num_inputs=num_inputs, 
                                           num_actions=action_space.shape[0], 
                                           hidden_dim=args.hidden_size,
                                           num_tasks=self.num_tasks,
                                           shared_feature_dim=args.shared_feature_dim,
                                           action_space=action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = EWCDeterministicPolicy(num_inputs=num_inputs, 
                                           num_actions=action_space.shape[0], 
                                           hidden_dim=args.hidden_size,
                                           num_tasks=self.num_tasks,
                                           shared_feature_dim=args.shared_feature_dim,
                                           action_space=action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, task_id, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state, task_id)
        else:
            _, _, action = self.policy.sample(state, task_id)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory_list, batch_size, updates, previous_task_id=None):
        # Sample a batch from memory
        current_memory = memory_list[self.task_id]
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = current_memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, self.task_id)
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            qf1_next_target, qf2_next_target = self.critic_targets[self.task_id](next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critics[self.task_id](state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        # self.critic_optim.zero_grad()
        self.critic_optims[self.task_id].zero_grad()
        qf_loss.backward()
        # self.critic_optim.step()
        self.critic_optims[self.task_id].step()

        pi, log_pi, _ = self.policy.sample(state_batch, self.task_id)

        qf1_pi, qf2_pi = self.critics[self.task_id](state_batch, pi)
        # qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        #TODO: Can we remove this part
        if self.task_id > 0:
            regularization_loss = 0
            for previous_task_id in range(self.task_id):
                previous_memory = memory_list[previous_task_id]
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = previous_memory.sample(batch_size=batch_size)
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

                previous_pi, previous_log_pi, _ = self.policy.sample(state_batch, previous_task_id)

                # qf1_pi, qf2_pi = self.critic(state_batch, previous_pi)
                qf1_pi, qf2_pi = self.critics[previous_task_id](state_batch, previous_pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                # previous_policy_loss = ((self.alpha * previous_log_pi) - min_qf_pi).mean()
                if self.args.algorithm == "EWC":
                    previous_policy_loss = (- min_qf_pi).mean()
                    shared_parameters = list(self.policy.shared_linear1.parameters()) + \
                                        list(self.policy.shared_linear2.parameters()) + \
                                        list(self.policy.shared_linear3.parameters())
                    grads = torch.autograd.grad(previous_policy_loss, shared_parameters)
                    grad_square = []
                    for grad in grads:
                        grad_square.append(grad ** 2)
                    regularization_loss += (grads[0] * (list(self.policy.shared_linear1.parameters())[0]\
                        - list(self.previous_policy[previous_task_id].shared_linear1.parameters())[0])).sum()
                elif self.args.algorithm == "L2":
                    regularization_loss += (list(self.policy.shared_linear1.parameters())[0]\
                        - list(self.previous_policy[previous_task_id].shared_linear1.parameters())[0]).sum()
                else:
                    raise ValueError("Error, choose the algorithm that was not implemented")

            total_loss = policy_loss + self.args.ewc_gamma * regularization_loss

            self.policy.zero_grad()
            self.policy_optim.zero_grad()
            total_loss.backward()
            
            
            
        elif self.task_id == 0:
            self.policy_optim.zero_grad()
            policy_loss.backward()

        # self.policy_optim.zero_grad()
        # policy_loss.backward()
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
            soft_update(self.critic_targets[self.task_id], self.critics[self.task_id], self.tau)
            # soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = os.path.join(self.outdir, "actor_{}.ckpt".format(suffix))
        if critic_path is None:
            # critic_path = os.path.join(self.outdir, "critic_{}.ckpt".format(suffix))
            critic_path = os.path.join(self.outdir, "critic.ckpt".format(suffix))
            # don't wanna save the critic
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        # torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.critics.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critics.load_state_dict(torch.load(critic_path))
    def set_task_id(self, task_id):
        self.task_id = task_id  
    def remember_prev_policy(self):
        self.previous_policy.append(copy.deepcopy(self.policy))
        # self.previous_policy.append(self.policy)
