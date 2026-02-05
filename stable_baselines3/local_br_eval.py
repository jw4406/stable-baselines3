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
parser.add_argument("--main_checkpoint_model_path", type=str, required=True)
parser.add_argument("--done_model_checkpoint_path", type=str, required=True)
parser.add_argument("--br_checkpoint_model_path", type=str, required=True)
parser.add_argument("--env_id", type=str, required=True)
parser.add_argument("--ego_strength", type=float, required=True)
parser.add_argument("--adv_strength", type=float, required=True)
args = parser.parse_args()
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
env.action_space = model.dstb_action_space
br_model = Exploiter.load(BR_MODEL_PATH, env=env, n_envs=1)
nr = 50 
rewards, selfplay_rewards = [], []
for i in range(nr):
    curr_reward = 0
    obs = model.env.reset()
    obs = np.expand_dims(obs, 0)
    done = False
    while not done:
        with th.no_grad():   
            action, _, _, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
            action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
        action = action.cpu().numpy()
        action_br = action_br.cpu().numpy()
        clipped_action = np.hstack([action, action_br])
        obs, reward, done, info = model.env.step(clipped_action)
        curr_reward += reward
    rewards.append(curr_reward)
    print(f"Episode {i+1} completed")

for i in range(nr):
    selfplay_reward = 0
    obs = model.env.reset()
    obs = np.expand_dims(obs, 0)
    done = False
    while not done:
        with th.no_grad():   
            action, _, _, adv_action, _, _ = model.policy(obs_as_tensor(obs, model.device))
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
with open(os.path.join(br_rewards_folder, "%s.txt" % str(model.num_timesteps)), "w") as f:
    f.write(str(np.mean(rewards)))
with open(os.path.join(selfplay_rewards_folder, "%s.txt" % str(model.num_timesteps)), "w") as f:
    f.write(str(np.mean(selfplay_rewards)))