import gymnasium as gym
from gymnasium.envs.registration import register
# Example for the CartPole environment
from stable_baselines3.a2c.my_pendulum import my_PendulumEnv
from stable_baselines3.a2c.my_walker2d_v4 import my_Walker2dEnv
from stable_baselines3.a2c.my_mountain_car_continuous import my_Continuous_MountainCarEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList
import argparse
from gymnasium.spaces import Box
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--jobid', default=None, required=False)
#parser.set_defaults(jobid=0)

args = parser.parse_args()
register(
    # unique identifier for the env `name-version`
    id="my_pendulum",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_PendulumEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=200,
)
register(# unique identifier for the env `name-version`
    id="my_walker2d_v4",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_Walker2dEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=1000,
)
register(# unique identifier for the env `name-version`
    id="my_mountain_car_continuous",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_Continuous_MountainCarEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=999,
)

register(# unique identifier for the env `name-version`
    id="my_half_cheetah",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_HalfCheetahEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=1000,
)

import stable_baselines3.a2c
from stable_baselines3 import A2C
from stable_baselines3 import A3C_rarl

#env = gym.make("HalfCheetah-v4")
env = gym.make("my_half_cheetah")
#env = gym.make("my_pendulum")
model = A2C("MlpPolicy", env, verbose=1, normalize_advantage=False,gae_lambda=.98,ent_coef=0.0,max_grad_norm=.9,n_steps=8,vf_coef=.6,gamma=.98,learning_rate=1e-3,use_sde=True,use_rms_prop=False)
#model = A2C("MlpPolicy", env=env, verbose=1, normalize_advantage=True, n_steps=100, use_sde=True, use_rms_prop=False)

model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (2,), dtype=np.float32), use_stackelberg=True, env=env, verbose=2, n_steps=512, normalize_advantage=False,gae_lambda=.92,ent_coef=0.0,max_grad_norm=.8,vf_coef=.4,gamma=.98,v_learning_rate=5e-3, c_learning_rate=1e-2,d_learning_rate=5e-2, use_sde=True,use_rms_prop=False)

#model = A3C_rarl("MlPAACPolicy", use_stackelberg=True,env=env, verbose=1, normalize_advantage=False, n_steps=8, v_learning_rate=5e-4, c_learning_rate=1e-3,d_learning_rate=5e-3, use_sde=True, use_rms_prop=False)
#model = A3C_rarl("MlPAACPolicy", use_stackelberg=True,env=env, verbose=1, normalize_advantage=True, n_steps=100, v_learning_rate=5e-4, c_learning_rate=1e-3,d_learning_rate=5e-3, use_sde=True, use_rms_prop=False)

#model = A3C_rarl.load("./logs/stac_pend_model_820000_steps.zip", env=env)
#model = A2C.load("mcc_0.zip", env=env)
#model = A3C_rarl.load("adv_pendulum_split_0.zip", env=env)
#model.v_learning_rate = 1e-6
#model.c_learning_rate = 1e-7
#model.d_learning_rate = 6e-7
#model.n_steps = 7
#model.use_stackelberg=True
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1, eval_freq=16, jobid=args.jobid)
checkpoint_callback = CheckpointCallback(
  save_freq=10,
  save_path="./logs/",
  name_prefix="cpu_cheetah_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
  jobid=args.jobid
)
callback_list = CallbackList([eval_callback, checkpoint_callback])
model.learn(total_timesteps=10_000_000, callback=callback_list)
#callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
#eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
#model.save("stac_pend_sanity.zip")
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, dstb_action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step([[action, dstb_action, i]])
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
