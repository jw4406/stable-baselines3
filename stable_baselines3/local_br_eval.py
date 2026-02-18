from stable_baselines3.main.common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.exploiter import Exploiter
from stable_baselines3.a2c.my_ant_v5 import my_AntEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from gymnasium import register
import gymnasium as gym
from os import getcwd as pwd
import os
import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
import imageio
import argparse
from stable_baselines3.a2c.my_ant_v5 import my_AntEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
from stable_baselines3.a2c.my_hopper_v5 import my_HopperEnv
from stable_baselines3.a2c.my_pendulum import my_PendulumEnv
from stable_baselines3.a2c.my_walker2d_v4 import my_Walker2dEnv
from stable_baselines3.a2c.my_mountain_car_continuous import my_Continuous_MountainCarEnv
parser = argparse.ArgumentParser()
parser.add_argument("--eval_prot", type=str, required=True)
parser.add_argument("--main_checkpoint_model_path", type=str, required=True)
parser.add_argument("--done_model_checkpoint_path", type=str, required=True)
parser.add_argument("--br_checkpoint_model_path", type=str, required=True)
parser.add_argument("--env_id", type=str, required=True)
parser.add_argument("--ego_strength", type=float, required=True)
parser.add_argument("--adv_strength", type=float, required=True)
parser.add_argument("--exploiter_is_cds", type=str, required=True)
parser.add_argument("--br_index", type=int, required=True)
args = parser.parse_args()
args.exploiter_is_cds = args.exploiter_is_cds == 'True'
MAIN_CHECKPOINT_MODEL_PATH = args.main_checkpoint_model_path
DONE_MODEL_CHECKPOINT_PATH = args.done_model_checkpoint_path
br_rewards_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "br_rewards")
selfplay_rewards_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfplay_rewards")
os.makedirs(br_rewards_folder, exist_ok=True)
os.makedirs(selfplay_rewards_folder, exist_ok=True)
BR_MODEL_PATH = args.br_checkpoint_model_path
ENV_ID = args.env_id
register(
    id="my_pendulum",
    entry_point=my_PendulumEnv,
    max_episode_steps=1000 if ENV_ID != "my_pendulum" else 200,
)
register(
    id="my_half_cheetah",
    entry_point=my_HalfCheetahEnv,
    max_episode_steps=1000 if ENV_ID != "my_half_cheetah" else 1000,
)
register(
    id="my_hopper",
    entry_point=my_HopperEnv,
    max_episode_steps=1000 if ENV_ID != "my_hopper" else 1000,
)
register(
    id="my_walker2d_v4",
    entry_point=my_Walker2dEnv,
    max_episode_steps=1000 if ENV_ID != "my_walker2d_v4" else 1000,
)
register(
    id="my_mountain_car_continuous",
    entry_point=my_Continuous_MountainCarEnv,
    max_episode_steps=999 if ENV_ID != "my_mountain_car_continuous" else 999,
)
register(
    id="my_ant",
    entry_point=my_AntEnv,
    max_episode_steps=1000 if ENV_ID != "my_ant" else 1000,
)
env = gym.make(ENV_ID)
try:
    model = CleanDerivativeFreeSPAR.load(MAIN_CHECKPOINT_MODEL_PATH, env=env, num_perturbed=1)
except FileNotFoundError:
    model = CleanDerivativeFreeSPAR.load(DONE_MODEL_CHECKPOINT_PATH, env=env, num_perturbed=1)
if args.exploiter_is_cds:
    pass
else:
    env.action_space = model.dstb_action_space
if args.exploiter_is_cds:
    br_model = CleanDerivativeFreeSPAR.load(BR_MODEL_PATH, env=env, num_perturbed=1)
else:
    br_model = Exploiter.load(BR_MODEL_PATH, env=env, n_envs=1)
nr = 50 
exploiting_ego_rewards, selfplay_rewards, exploiting_adv_rewards = [], [], []
for i in range(nr):
    curr_reward = 0
    obs = model.env.reset()
    obs = np.expand_dims(obs, 0)
    done = False
    while not done:
        with th.no_grad():   
            action, _, adv_action, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
            if args.exploiter_is_cds:
                left_br_action, left_br_log_probs, right_br_action, right_br_log_probs, values, q_values = br_model.policy(obs_as_tensor(obs, br_model.device), deterministic=False, ego_forward=True, adv_forward=True, zero_ego_action=False, zero_adv_action=True)
                if args.eval_prot:
                    action_br = right_br_action
                else:
                    action_br = left_br_action
            else:
                action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
        action = action.cpu().numpy()
        action_br = action_br.cpu().numpy()
        if args.eval_prot:
            clipped_action = np.hstack([action, action_br])
        else:
            clipped_action = np.hstack([action_br, action])
        obs, reward, done, info = model.env.step(clipped_action)
        curr_reward += reward
    exploiting_ego_rewards.append(curr_reward)
    print(f"Episode {i+1} completed")
# for i in range(nr):
#     curr_reward = 0
#     obs = model.env.reset()
#     obs = np.expand_dims(obs, 0)
#     done = False
#     while not done:
#         with th.no_grad():
#             if args.exploiter_is_cds:
#                 ego_actions, ego_log_probs, action_br, adv_log_probs, values, q_values = br_model.policy(obs_as_tensor(obs, br_model.device), deterministic=False, ego_forward=True, adv_forward=True, zero_ego_action=False, zero_adv_action=True)
#             else:
#                 action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
#             action, _, adv_action, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
#         action = action.cpu().numpy()

#         clipped_action = np.hstack([action_br, adv_action])
#         obs, reward, done, info = model.env.step(clipped_action)
#         curr_reward += reward
#     exploiting_adv_rewards.append(curr_reward)
#     print(f"Episode {i+1} completed")

for i in range(nr):
    selfplay_reward = 0
    obs = model.env.reset()
    obs = np.expand_dims(obs, 0)
    done = False
    while not done:
        with th.no_grad():   
            action, _, adv_action, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
            #action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
        action = action.cpu().numpy()
        adv_action = adv_action.cpu().numpy()
        clipped_action = np.hstack([action, adv_action])
        obs, reward, done, info = model.env.step(clipped_action)
        selfplay_reward += reward
    selfplay_rewards.append(selfplay_reward)
    print(f"Episode {i+1} completed")
# TODO: write out to a file and then aggregate the results and plot
working_dir = pwd()
#os.makedirs(rewards_folder, exist_ok=True)
with open(os.path.join(br_rewards_folder, "%s_br%d_ego.txt" % (str(model.num_timesteps), args.br_index)), "w") as f:
    f.write(str(np.mean(exploiting_ego_rewards)))
with open(os.path.join(br_rewards_folder, "%s_br%d_adv.txt" % (str(model.num_timesteps), args.br_index)), "w") as f:
    f.write(str(np.mean(exploiting_adv_rewards)))
with open(os.path.join(selfplay_rewards_folder, "%s_br%d.txt" % (str(model.num_timesteps), args.br_index)), "w") as f:
    f.write(str(np.mean(selfplay_rewards)))