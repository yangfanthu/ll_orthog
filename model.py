import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import utils
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.constant_(m.bias, 0)



class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class LLQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_tasks):
        super(LLQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.ModuleList()
        for i in range(num_tasks):
            self.linear3.append(nn.Linear(hidden_dim, 1))

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.ModuleList()
        for i in range(num_tasks):
            self.linear6.append(nn.Linear(hidden_dim, 1))

        self.apply(weights_init_)

    def forward(self, state, action, task_id):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3[task_id](x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6[task_id](x2)

        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class LLGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_tasks, shared_feature_dim=256, action_space=None):
        super(LLGaussianPolicy, self).__init__()
        projections = utils.generate_projection_matrix(num_tasks=num_tasks, feature_dim=shared_feature_dim)
        utils.unit_test_projection_matrices(projections)
        self.projections = []
        for i in range(len(projections)):
            self.projections.append(torch.from_numpy(projections[i]).float())

        # TODO: check whether w should use bias here
        self.shared_linear1 = nn.Linear(num_inputs, hidden_dim, bias=False)
        self.shared_linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.shared_linear3 = nn.Linear(hidden_dim, shared_feature_dim, bias=False)

        self.mean_linears = nn.ModuleList()
        self.log_std_linears = nn.ModuleList()
        for i in range(num_tasks):
            self.mean_linears.append(nn.Linear(shared_feature_dim, num_actions))
            self.log_std_linears.append(nn.Linear(shared_feature_dim, num_actions))

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, task_id):
        x = F.relu(self.shared_linear1(state))
        x = F.relu(self.shared_linear2(x))
        x = F.relu(self.shared_linear3(x))
        projection = self.projections[task_id]
        x = torch.matmul(x,projection)

        mean_linear = self.mean_linears[task_id]
        log_std_linear = self.log_std_linears[task_id]
        mean = mean_linear(x)
        log_std = log_std_linear(x)

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, task_id):
        mean, log_std = self.forward(state, task_id)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        for i in range(len(self.projections)):
            self.projections[i] = self.projections[i].to(device)
        return super(LLGaussianPolicy, self).to(device)
    def zero_grad(self):
        self.single_zero_grad(self.shared_linear1.weight)        
        self.single_zero_grad(self.shared_linear2.weight)
        self.single_zero_grad(self.shared_linear3.weight)
        for module in self.mean_linears:
            self.single_zero_grad(module.weight)
        for module in self.log_std_linears:
            self.single_zero_grad(module.weight)

    def single_zero_grad(self, p):
        if p.grad is not None:
            if p.grad.grad_fn is not None:
                p.grad.detach_()
            else:
                p.grad.requires_grad_(False)
            p.grad.zero_()

class EWCGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_tasks, shared_feature_dim=256, action_space=None):
        super(EWCGaussianPolicy, self).__init__()
        # TODO: check whether w should use bias here
        self.shared_linear1 = nn.Linear(num_inputs, hidden_dim, bias=False)
        self.shared_linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.shared_linear3 = nn.Linear(hidden_dim, shared_feature_dim, bias=False)

        self.mean_linears = nn.ModuleList()
        self.log_std_linears = nn.ModuleList()
        for i in range(num_tasks):
            self.mean_linears.append(nn.Linear(shared_feature_dim, num_actions))
            self.log_std_linears.append(nn.Linear(shared_feature_dim, num_actions))

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, task_id):
        x = F.relu(self.shared_linear1(state))
        x = F.relu(self.shared_linear2(x))
        x = F.relu(self.shared_linear3(x))

        mean_linear = self.mean_linears[task_id]
        log_std_linear = self.log_std_linears[task_id]
        mean = mean_linear(x)
        log_std = log_std_linear(x)

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, task_id):
        mean, log_std = self.forward(state, task_id)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(EWCGaussianPolicy, self).to(device)
    def zero_grad(self):
        self.single_zero_grad(self.shared_linear1.weight)        
        self.single_zero_grad(self.shared_linear2.weight)
        self.single_zero_grad(self.shared_linear3.weight)
        for module in self.mean_linears:
            self.single_zero_grad(module.weight)
        for module in self.log_std_linears:
            self.single_zero_grad(module.weight)

    def single_zero_grad(self, p):
        if p.grad is not None:
            if p.grad.grad_fn is not None:
                p.grad.detach_()
            else:
                p.grad.requires_grad_(False)
            p.grad.zero_()
if __name__ == "__main__":
    policy = EWCGaussianPolicy(num_inputs=1, num_actions=1, hidden_dim=256, num_tasks=4)
    input = np.array([[1],[2]], dtype=np.float32)
    input = torch.from_numpy(input)
    action, log_prob, mean = policy.sample(input,0)
    gt = np.array([[1],[0]], dtype=np.float32)
    gt = torch.from_numpy(gt)
    criterion = nn.MSELoss()
    loss = criterion(gt, action)
    loss.backward()

    policy.zero_grad()
