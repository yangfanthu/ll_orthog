import os
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import robel
from sac import LLSAC, SAC
from ewc import EWCSAC
from gem import GEMSAC
from agem import AGEMSAC
from er import ERSAC
from apd import APDSAC
import random
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
parser.add_argument("--action-noise-scale", type=float, default=0.)
parser.add_argument("--algorithm", type=str, default='LL',
                    help="LL or EWC or L2 or GEM or AGEM or ER or SAC")
parser.add_argument('--ewc-gamma', type=float, default=1e-2,
                    help =" the ewc temperature of the previous tasks parameters")
parser.add_argument('--learn-critic', type=bool, default=False,
                    help='whether use lifelong leanring algorithm for critic learning')
parser.add_argument("--bias-weight", type=float, default=1e-2)
parser.add_argument("--diff-weight", type=float, default=3e-4)
args = parser.parse_args()
args.cuda = True if args.cuda and torch.cuda.is_available() else False
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
                # 'DClawTurnFixedT5-v0','DClawTurnFixedT6-v0','DClawTurnFixedT7-v0','DClawTurnFixedT8-v0','DClawTurnFixedT9-v0']
env_name_list = ['DClawTurnFixedT9-v0','DClawTurnFixedT8-v0','DClawTurnFixedT7-v0','DClawTurnFixedT6-v0','DClawTurnFixedT5-v0',
                'DClawTurnFixedT4-v0','DClawTurnFixedT3-v0','DClawTurnFixedT2-v0','DClawTurnFixedT1-v0','DClawTurnFixedT0-v0']

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


#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                             args.algorithm))
# save directory
if not os.path.exists('saved_models'):
    os.system('mkdir saved_models')
outdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = os.path.join('./saved_models', outdir)
os.system('mkdir ' + outdir)
with open(outdir+'/setting.txt','w') as f:
    # f.writelines("lifelong learning on only turning tasks real only on turning tasks\n")
    f.writelines("use episode 1200 and 1500 as the basic requirement of breaking, to test whether we need to train long enough to the next task\n")
    # f.writelines("remove alpha entropy loss for the previous tasks\n")
    f.writelines("order b\n")
    # f.writelines("LL without addtional sample\n")
    
    for each_arg, value in args.__dict__.items():
        f.writelines(each_arg + " : " + str(value)+"\n")

# Agent
if args.algorithm == "LL":
    agent = LLSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir)
elif args.algorithm == "EWC" or args.algorithm == "L2":
    agent = EWCSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir)
elif args.algorithm == "GEM":
    agent = GEMSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir)
elif args.algorithm == "AGEM":
    agent = AGEMSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir)
elif args.algorithm == "ER":
    agent = ERSAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir)
elif args.algorithm == "APD":
    agent = APDSAC(env.observation_space.shape[0],env.action_space,args=args, outdir=outdir)
elif args.algorithm == "SAC":
    agent = SAC(env.observation_space.shape[0], env.action_space, num_tasks, args, outdir)
# Training Loop
total_numsteps = 0

updates = 0
for task_id, env_name in enumerate(env_name_list):
    current_task_numsteps = 0
    print("the current training env is {}".format(env_name))
    env = gym.make(env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    action_space_shape = env.action_space.shape
    state = env.reset()
    agent.set_task_id(task_id)
    if task_id > 0 and (args.algorithm == "EWC" or args.algorithm == "L2"):
        agent.remember_prev_policy()
    if args.algorithm == "APD":
        agent.add_task()
    if args.algorithm == "APD" and task_id > 0:
        #clean the buffer
        memory_list[task_id-1].buffer = []
        memory_list[task_id-1].position = 0
    agent.alpha = args.alpha
    best_reward = -99999
    for i_episode in range(args.training_episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > current_task_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state, task_id=task_id)  # Sample action from policy

            if len(memory_list[task_id]) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory_list, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            noise = np.random.randn(action_space_shape[0])
            noise = noise * args.action_noise_scale
            input_action = action + noise
            next_state, reward, done, _ = env.step(input_action) # Step
            episode_steps += 1
            total_numsteps += 1
            current_task_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory_list[task_id].push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        if current_task_numsteps > args.num_steps:
            current_task_numsteps = 0
            break
        if i_episode % 10 == 0:    
            writer.add_scalars('reward/train', {env_name: episode_reward}, i_episode)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 30 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 1
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, task_id = task_id, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalars('avg_reward/test', {env_name: avg_reward}, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(suffix=total_numsteps)
        if i_episode % 50 == 0:
            agent.save_model(suffix=total_numsteps)
        # TODO: do we need to break in this way?
        # if env_name != env_name_list[-1]:
        #     if "Turn" in env_name:
        #         # if avg_reward > 2800:
        #         if avg_reward > 2800 and i_episode > 2000:
        #             break
        #     elif "Grab" in env_name:
        #         # if avg_reward >5000:
        #         if avg_reward > 5000 and i_episode > 1500:
        #             break
        #     else:
        #         print("error")
        # else:
        #     fail_task_list = test(agent)
        #     if "Turn" in env_name:
        #         # if avg_reward > 2800:
        #         if avg_reward > 2800 and i_episode > 2000:
        #             if len(fail_task_list) == 0:
        #                 break
        #             else:
        #                 prev_index = random.randint(0, len(fail_task_list) - 1)
        #                 agent.update_parameters(memory_list, args.batch_size, updates, fail_task_list[prev_index])
        #     elif "Grab" in env_name:
        #         # if avg_reward >5000:
        #         if avg_reward > 5000 and i_episode > 1500:
        #             if len(fail_task_list) == 0:
        #                 break
        #             else:
        #                 prev_index = random.randint(0, len(fail_task_list) - 1)
        #                 agent.update_parameters(memory_list, args.batch_size, updates, fail_task_list[prev_index])
        # if "Turn" in env_name:
        #     # if avg_reward > 2800:
        #     # if avg_reward > 3000 and i_episode > 3000:
        #     if avg_reward > 3000 and i_episode > 1000:
        #         break
        # elif "Grab" in env_name:
        #     # if avg_reward >5000:
        #     # if avg_reward > 5000 and i_episode > 3500:
        #     if avg_reward > 5000 and i_episode > 1000:
        #         break
        # else:
        #     print("error")
        if "Turn" in env_name:
            # if avg_reward > 2800:
            # if avg_reward > 3000 and i_episode > 3000:
            if avg_reward > 900 and i_episode > 2000:
                break
        else:
            print("error")
        
    agent.save_model(suffix=total_numsteps)
    env.close()


