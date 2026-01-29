from stable_baselines3.main.common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.exploiter import Exploiter
from stable_baselines3.a2c.my_ant_v5 import my_AntEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from gymnasium import register
import gymnasium as gym
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
parser.add_argument("--br_checkpoint_model_path", type=str, required=True)
parser.add_argument("--env_id", type=str, required=True)
parser.add_argument("--ego_strength", type=float, required=True)
parser.add_argument("--adv_strength", type=float, required=True)
args = parser.parse_args()
MAIN_CHECKPOINT_MODEL_PATH = args.main_checkpoint_model_path
BR_MODEL_PATH = args.br_checkpoint_model_path
ENV_ID = args.env_id
register(
    id=ENV_ID,
    entry_point=globals()[f"my_{ENV_ID}"],
    max_episode_steps=1000 if ENV_ID != "my_pendulum" else 200,
)
model = CleanDerivativeFreeSPAR.load(MAIN_CHECKPOINT_MODEL_PATH)
br_model = Exploiter.load(BR_MODEL_PATH)
env = gym.make(ENV_ID)
nr = 5
rewards = []
for i in range(nr):
    curr_reward = 0
    obs = env.reset()
    done = False
    while not done:
        action, _, _, _, _, _ = model.policy(obs_as_tensor(obs, model.device))
        action_br, _, _ = br_model.policy(obs_as_tensor(obs, br_model.device))
        clipped_action = np.hstack([action, action_br])
        obs, reward, done, info = env.step(clipped_action)
        curr_reward += reward
    rewards.append(curr_reward)
    print(f"Episode {i+1} completed")
