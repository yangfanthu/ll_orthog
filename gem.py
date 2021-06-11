import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import EWCGaussianPolicy, QNetwork, DeterministicPolicy, LLQNetwork
import random
import quadprog
import itertools

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1
def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

class GEMSAC(object):
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
        self.gpu = True if args.cuda else False
        
        self.task_id = 0
        if self.args.learn_critic:
            self.critic = LLQNetwork(num_inputs, action_space.shape[0], args.hidden_size, num_tasks=num_tasks).to(device=self.device)
            self.critic_target = LLQNetwork(num_inputs, action_space.shape[0], args.hidden_size, num_tasks=num_tasks).to(device=self.device)
            self.first_critic_optim = Adam(self.critic.parameters(), lr = args.lr)
            optim_critic_parameters = list(self.critic.linear3.parameters()) + list(self.critic.linear6.parameters())
            self.critic_optim = Adam(optim_critic_parameters, lr = args.lr)

            hard_update(self.critic, self.critic_target)

            shared_critic_parameters = list(self.critic.linear1.parameters()) + \
                                list(self.critic.linear2.parameters()) + \
                                list(self.critic.linear4.parameters()) + \
                                list(self.critic.linear5.parameters())
            self.critic_grad_dims = []
            for param in shared_critic_parameters:
                self.critic_grad_dims.append(param.data.numel())
            self.critic_grads = torch.Tensor(sum(self.critic_grad_dims), num_tasks)
            if args.cuda:
                self.critic_grads = self.critic_grads.cuda()
        else:
            self.critics = nn.ModuleList()
            self.critic_optims = []
            self.critic_targets = nn.ModuleList()
            
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
            optim_policy_parameters = list(self.policy.mean_linears.parameters()) + list(self.policy.log_std_linears.parameters())
            self.first_policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            self.policy_optim = Adam(optim_policy_parameters, lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        shared_parameters = list(self.policy.shared_linear1.parameters()) + \
                                list(self.policy.shared_linear2.parameters()) + \
                                list(self.policy.shared_linear3.parameters())
        # shared_parameters = itertools.chain(self.policy.shared_linear1.parameters(), self.policy.shared_linear2.parameters(), self.policy.shared_linear3.parameters())
        self.grad_dims = []
        for param in shared_parameters:
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), num_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

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
            if self.args.learn_critic:
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, self.task_id)
            else:
                qf1_next_target, qf2_next_target = self.critic_targets[self.task_id](next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        if self.args.learn_critic:
            qf1, qf2 = self.critic(state_batch, action_batch, self.task_id)
        else:
            qf1, qf2 = self.critics[self.task_id](state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        if self.task_id > 0 and self.args.learn_critic:
            for previous_task_id in range(self.task_id):
                self.critic.zero_grad()
                previous_memory = memory_list[previous_task_id]
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = previous_memory.sample(batch_size=batch_size)
                
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, self.task_id)
                    # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                    prev_qf1_next_target, prev_qf2_next_target = self.critic_target(next_state_batch, next_state_action, self.task_id)
                    prev_min_qf_next_target = torch.min(prev_qf1_next_target, prev_qf2_next_target) - self.alpha * next_state_log_pi
                    prev_next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
                # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
                prev_qf1, prev_qf2 = self.critic(state_batch, action_batch, self.task_id)
                prev_qf1_loss = F.mse_loss(prev_qf1, prev_next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                prev_qf2_loss = F.mse_loss(prev_qf2, prev_next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                prev_qf_loss = prev_qf1_loss + prev_qf2_loss
                shared_critic_parameters = list(self.critic.linear1.parameters()) + \
                                list(self.critic.linear2.parameters()) + \
                                list(self.critic.linear4.parameters()) + \
                                list(self.critic.linear5.parameters())
                torch.autograd.grad(prev_qf_loss, shared_critic_parameters)
                store_grad(shared_critic_parameters, self.critic_grads, self.critic_grad_dims, previous_task_id)
            self.critic.zero_grad()
            self.critic_optim.zero_grad()
            qf_loss.backward()
            shared_critic_parameters = list(self.critic.linear1.parameters()) + \
                                list(self.critic.linear2.parameters()) + \
                                list(self.critic.linear4.parameters()) + \
                                list(self.critic.linear5.parameters())
            store_grad(shared_critic_parameters, self.critic_grads, self.critic_grad_dims, self.task_id)
            indx = torch.cuda.LongTensor(range(self.task_id)) if self.gpu \
                else torch.LongTensor(range(self.task_id))
            dotp = torch.mm(self.critic_grads[:, self.task_id].unsqueeze(0),
                            self.critic_grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.critic_grads[:, self.task_id].unsqueeze(1),
                                self.critic_grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(shared_critic_parameters, self.critic_grads[:, self.task_id],
                                self.critic_grad_dims)
    
        if self.args.learn_critic:
            if self.task_id == 0:
                self.first_critic_optim.zero_grad()
                qf_loss.backward()
                self.first_critic_optim.step()
            else:
                self.critic_optim.step()
        else:
            self.critic_optims[self.task_id].zero_grad()
            qf_loss.backward()
            self.critic_optims[self.task_id].step()
        # self.critic_optim.step()
        

        pi, log_pi, _ = self.policy.sample(state_batch, self.task_id)

        if self.args.learn_critic:
            qf1_pi, qf2_pi = self.critic(state_batch, pi, self.task_id)
        else:
            qf1_pi, qf2_pi = self.critics[self.task_id](state_batch, pi)
        # qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        #TODO: Can we remove this part
        if self.task_id > 0:
            for previous_task_id in range(self.task_id):
                self.policy.zero_grad()
                previous_memory = memory_list[previous_task_id]
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = previous_memory.sample(batch_size=batch_size)
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

                previous_pi, previous_log_pi, _ = self.policy.sample(state_batch, previous_task_id)

                # qf1_pi, qf2_pi = self.critic(state_batch, previous_pi)
                if self.args.learn_critic:
                    qf1_pi, qf2_pi = self.critic(state_batch, previous_pi, self.task_id)
                else:
                    qf1_pi, qf2_pi = self.critics[previous_task_id](state_batch, previous_pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                # previous_policy_loss = ((self.alpha * previous_log_pi) - min_qf_pi).mean()
                previous_policy_loss = (- min_qf_pi).mean()
                shared_parameters = list(self.policy.shared_linear1.parameters()) + \
                                    list(self.policy.shared_linear2.parameters()) + \
                                    list(self.policy.shared_linear3.parameters())
                # shared_parameters = itertools.chain(self.policy.shared_linear1.parameters(), self.policy.shared_linear2.parameters(), self.policy.shared_linear3.parameters())
                torch.autograd.grad(previous_policy_loss, shared_parameters)
                store_grad(shared_parameters, self.grads, self.grad_dims, previous_task_id)



            self.policy.zero_grad()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            shared_parameters = list(self.policy.shared_linear1.parameters()) + \
                                list(self.policy.shared_linear2.parameters()) + \
                                list(self.policy.shared_linear3.parameters())
            # shared_parameters = itertools.chain(self.policy.shared_linear1.parameters(), self.policy.shared_linear2.parameters(), self.policy.shared_linear3.parameters())
            # check if gradient violates constraints
            # copy gradient
            store_grad(shared_parameters, self.grads, self.grad_dims, self.task_id)
            indx = torch.cuda.LongTensor(range(self.task_id)) if self.gpu \
                else torch.LongTensor(range(self.task_id))
            dotp = torch.mm(self.grads[:, self.task_id].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, self.task_id].unsqueeze(1),
                                self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(shared_parameters, self.grads[:, self.task_id],
                                self.grad_dims)
            # for param in shared_parameters:
            #     print(param.grad.data)
            #     print(list(self.policy.shared_linear1.parameters())[0].grad.data)
            
            
        elif self.task_id == 0:
            self.first_policy_optim.zero_grad()
            policy_loss.backward()

        # self.policy_optim.zero_grad()
        # policy_loss.backward()
        if self.task_id == 0:
            self.first_policy_optim.step()
        else:
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
            if self.args.learn_critic:
                soft_update(self.critic_target, self.critic, self.tau)
            else:
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
        if self.args.learn_critic:
            torch.save(self.critic.state_dict(), critic_path)
        else:
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
if __name__ == "__main__":
    import gym
    import numpy as np
    import torch
    import robel
    import random
    import argparse
    from torch.utils.tensorboard import SummaryWriter
    from replay_memory import ReplayMemory

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='run on CUDA (default: True)')
    # parser.add_argument('--cuda', action="store_true",
    #                     help='run on CUDA (default: False)')

    parser.add_argument("--training-episodes", type=int, default=int(5e3), 
    # parser.add_argument("--training-episodes", type=int, default=int(1e2), 
                        help="num of maximum episodes for training each tasks")
    parser.add_argument("--shared-feature-dim", type=int, default=512,
                        help="the feature dim of the shared feature in the policy network")
    parser.add_argument('--ewc-gamma', type=float, default=1e-2,
                        help =" the ewc temperature of the previous tasks parameters")

    args = parser.parse_args()

    # env_name_list = ['DClawTurnFixedF3-v0','DClawTurnFixedF1-v0','DClawTurnFixedF2-v0','DClawTurnFixedF0-v0','DClawTurnFixedF4-v0']
    # env_name_list = ['DClawTurnFixedF3-v0','DClawTurnFixedF1-v0','DClawTurnFixedF2-v0','DClawTurnFixedF0-v0','DClawTurnFixedF4-v0',
    #                 'DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0']
    # env_name_list = ['DClawTurnFixedF3-v0','DClawTurnFixedF1-v0','DClawTurnFixedF2-v0','DClawTurnFixedF0-v0','DClawTurnFixedF4-v0',
    #                 'DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0', 'DClawGrabFixedFF0-v0', 'DClawGrabFixedFF1-v0']
    # env_name_list = ['DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0', 'DClawGrabFixedFF0-v0', 'DClawGrabFixedFF1-v0']
    env_name_list = ['DClawTurnFixedF3-v0','DClawTurnFixedF1-v0','DClawTurnFixedF2-v0','DClawTurnFixedF0-v0','DClawTurnFixedF4-v0',
                    'DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0', 'DClawGrabFixedFF0-v0']
    # env_name_list = ['DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0', 'DClawGrabFixedFF0-v0']

    num_tasks = len(env_name_list)
    memory_list = []
    for i in range(len(env_name_list)):
        memory_list.append(ReplayMemory(args.replay_size, args.seed))

    # action space of different tasks is assumed to be the same
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(env_name_list[0])
    env.seed(args.seed)
    random.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # Agent
    agent = GEMSAC(env.observation_space.shape[0], env.action_space, num_tasks, args)
