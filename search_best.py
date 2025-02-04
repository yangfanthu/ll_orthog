import os
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import robel
import tqdm
from sac import LLSAC, SAC
from ewc import EWCSAC
from gem import GEMSAC
from apd import APDSAC
from er import ERSAC
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
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument("--training-episodes", type=int, default=int(3e3), 
                    help="num of maximum episodes for training each tasks")
parser.add_argument("--shared-feature-dim", type=int, default=512,
                    help="the feature dim of the shared feature in the policy network")
parser.add_argument("--algorithm", type=str, default='LL',
                    help="LL or EWC")
parser.add_argument('--learn-critic', type=bool, default=False,
                    help='whether use lifelong leanring algorithm for critic learning')
parser.add_argument("--bias-weight", type=float, default=1e-2)
parser.add_argument("--diff-weight", type=float, default=3e-4)
args = parser.parse_args()


# env_name_list = ['DClawTurnFixedF3-v0','DClawTurnFixedF1-v0','DClawTurnFixedF2-v0','DClawTurnFixedF0-v0','DClawTurnFixedF4-v0']
# env_name_list = ['DClawTurnFixedF3-v0','DClawTurnFixedF1-v0','DClawTurnFixedF2-v0','DClawTurnFixedF0-v0','DClawTurnFixedF4-v0',
#                 'DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0']
# env_name_list = ['DClawTurnFixedF3-v0','DClawTurnFixedF1-v0','DClawTurnFixedF2-v0','DClawTurnFixedF0-v0','DClawTurnFixedF4-v0',
#                 'DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0', 'DClawGrabFixedFF0-v0', 'DClawGrabFixedFF1-v0']
# env_name_list = ['DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0', 'DClawGrabFixedFF0-v0', 'DClawGrabFixedFF1-v0']
# env_name_list = ['DClawTurnFixedF3-v0','DClawTurnFixedF1-v0','DClawTurnFixedF2-v0','DClawTurnFixedF0-v0','DClawTurnFixedF4-v0',
#                 'DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0', 'DClawGrabFixedFF0-v0']
# env_name_list = ['DClawGrabFixedFF2-v0','DClawGrabFixedFF3-v0', 'DClawGrabFixedFF4-v0', 'DClawGrabFixedFF0-v0']
# env_name_list = ['DClawTurnFixedT0-v0','DClawTurnFixedT1-v0','DClawTurnFixedT2-v0','DClawTurnFixedT3-v0','DClawTurnFixedT4-v0',
#                 'DClawTurnFixedT5-v0','DClawTurnFixedT6-v0','DClawTurnFixedT7-v0','DClawTurnFixedT8-v0','DClawTurnFixedT9-v0']
env_name_list = ['DClawTurnFixedT9-v0','DClawTurnFixedT8-v0','DClawTurnFixedT7-v0','DClawTurnFixedT6-v0','DClawTurnFixedT5-v0',
                'DClawTurnFixedT4-v0','DClawTurnFixedT3-v0','DClawTurnFixedT2-v0','DClawTurnFixedT1-v0','DClawTurnFixedT0-v0']

# env_name_list = ['DClawTurnFixedT3-v0','DClawTurnFixedT1-v0','DClawTurnFixedT7-v0','DClawTurnFixedT4-v0','DClawTurnFixedT2-v0',
#                 'DClawTurnFixedT0-v0','DClawTurnFixedT8-v0','DClawTurnFixedT9-v0','DClawTurnFixedT5-v0','DClawTurnFixedT6-v0']

num_tasks = len(env_name_list)

# action space of different tasks is assumed to be the same
# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(env_name_list[0])
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
if args.algorithm == "LL":
    agent = LLSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir=None)
elif args.algorithm == "EWC" or args.algorithm == "L2":
    agent = EWCSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir=None)
elif args.algorithm == "GEM" or args.algorithm == "AGEM":
    agent = GEMSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir=None)
elif args.algorithm == "ER":
    agent = ERSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir=None)
elif args.algorithm == "APD":
    agent = APDSAC(env.observation_space.shape[0],env.action_space,args=args, outdir=None)
    for i in range(len(env_name_list)):
        agent.add_task()
elif args.algorithm == "SAC":
    agent = SAC(env.observation_space.shape[0], env.action_space, num_tasks, args, None)
policy_dir = './saved_models/2021-06-18_14-13-51'
start_index = 750000
end_index = 4000000
model_params = os.listdir(policy_dir)
model_candidates = []
for i, model_param in enumerate(model_params):
    if "critic" in model_param or "setting" in model_param:
        continue
    temp = model_param.split('_')
    temp = temp[1]
    temp = temp.split('.')
    temp = temp[0] 
    index = int(temp)
    if index > start_index and index < end_index:
        path = os.path.join(policy_dir, model_param)
        model_candidates.append(path)
# agent.policy.load_state_dict(torch.load('./saved_models/2021-06-07_09-54-33/actor_2322160.ckpt'))  



# Memory
# memory = ReplayMemory(args.replay_size, args.seed)
best_model = None
best_reward = -9999999
best_reward_list = []
# Training Loop
for model in tqdm.tqdm(model_candidates):
    agent.policy.load_state_dict(torch.load(model))
    reward_list = [] 
    for task_id, env_name in enumerate(env_name_list):
        env = gym.make(env_name)
        agent.set_task_id(task_id)
        # agent.alpha = args.alpha

        avg_reward = 0.
        episodes = 1
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            # env.render()
            while not done:
                action = agent.select_action(state, task_id = task_id, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                # env.render()

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes
        reward_list.append(avg_reward)

        # print("----------------------------------------")
        # print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        # print("----------------------------------------")
        env.close()
    reward_list = np.array(reward_list)
    print(best_reward)
    # if (reward_list > 0).all(): # the basic requirement, all the reward should be positive
    sum_reward = reward_list.sum()
    if sum_reward > best_reward:
        best_reward = sum_reward
        best_reward_list = reward_list
        best_model = model
print("________________")
print(best_reward_list)
print(best_model)

