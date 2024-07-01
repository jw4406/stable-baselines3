import gymnasium as gym
from gymnasium.envs.registration import register
# Example for the CartPole environment
from stable_baselines3.a2c.my_pendulum import my_PendulumEnv
from stable_baselines3.a2c.my_walker2d_v4 import my_Walker2dEnv
from stable_baselines3.a2c.my_mountain_car_continuous import my_Continuous_MountainCarEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
from stable_baselines3 import SAC, SMART
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList
import argparse
from multiprocessing import Pool
import os
from gymnasium.spaces import Box
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--jobid', default=None, required=False)
#parser.set_defaults(jobid=0)
def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
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
#from stable_baselines3 import SAC


#env = gym.make("MountainCarContinuous-v0")
#env = gym.make("my_half_cheetah", render_mode='human')
env = gym.make("my_pendulum")
v_learning_rate = 5e-4

tau_v_c = 2
tau_c_d = 5

#model = A3C_rarl("MlPAACPolicy", use_stackelberg=False, env=env, verbose=2, n_steps=8, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,vf_coef=.4,gamma=.9,v_learning_rate=5e-4, c_learning_rate=5e-4,d_learning_rate=5e-4, use_sde=True,use_rms_prop=False, device='cpu')

#model = lambda tau1, tau2: A3C_rarl("MlPAACPolicy", use_stackelberg=False, env=env, verbose=2, n_steps=8, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,vf_coef=.4,gamma=.9,v_learning_rate=v_learning_rate, c_learning_rate=v_learning_rate * tau1,d_learning_rate=v_learning_rate * tau1 * tau2, use_sde=True,use_rms_prop=False, device='cpu')
def f(tau2):
    seeds = [3721, 1234785, 834981, 9274, 42069, 92048, 109475, 373095, 5, 92038]
    model = A3C_rarl("MlPAACPolicy", use_stackelberg=True, env=env, verbose=2, n_steps=8, normalize_advantage=False,
                     gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,vf_coef=.4,gamma=.9,
                     v_learning_rate=linear_schedule(v_learning_rate),
                     c_learning_rate=linear_schedule(v_learning_rate * tau_v_c),
                     d_learning_rate=linear_schedule(v_learning_rate * tau_v_c * tau_c_d),
                     use_sde=True,use_rms_prop=False, device='auto', seed=seeds[int(tau2)])
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-150, verbose=1)

    #model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (2,), dtype=np.float32), use_stackelberg=True,
                     #env=env, verbose=2, n_steps=32, normalize_advantage=False, gae_lambda=.95, ent_coef=0.0,
                     #max_grad_norm=.7, vf_coef=.4, gamma=.95, v_learning_rate=linear_schedule(5e-4),
                     #c_learning_rate=linear_schedule(1e-3), d_learning_rate=linear_schedule(5e-3), use_sde=True,
                     #use_rms_prop=False)
    eval_callback = EvalCallback(env, verbose=1, callback_on_new_best=callback_on_best,n_eval_episodes=50, jobid=args.jobid)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./competitive_models/",
        #stac_train_sweep_pend_competitive_%d % int(tau2)
        name_prefix="baseline_train_pend_parallel_competitive_wd_53_ud_55_%d" % int(tau2),
        save_replay_buffer=True,
        save_vecnormalize=True,
        jobid=args.jobid
    )
    callback_list = CallbackList([eval_callback, checkpoint_callback])  # , checkpoint_callback])
    # model.learn(total_timesteps=1_000_000, callback=callback_list)
    model.learn(total_timesteps=7_500_000, callback=callback_list)
    model.save("./competitive_models/baseline_train_pend_parallel_FINISHED_wd_53_ud_55_%d.zip" % int(tau2))
    print("HI IM DONE")
if __name__ == '__main__':

    with Pool(10) as p:
        p.map(f, np.arange(0,10))


'''
#model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (2,), dtype=np.float32), use_stackelberg=True, env=env, verbose=2, n_steps=512, normalize_advantage=False,gae_lambda=.92,ent_coef=0.0,max_grad_norm=.8,vf_coef=.4,gamma=.98,v_learning_rate=5e-4, c_learning_rate=1e-3,d_learning_rate=5e-3, use_sde=True,use_rms_prop=False)
model=A2C("MlpPolicy", normalize_advantage=True, verbose=2, env=env, n_steps=100, learning_rate=linear_schedule(3e-4), use_sde=True, use_rms_prop=False)
#model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (2,), dtype=np.float32), use_stackelberg=True, env=env, verbose=2, n_steps=512, normalize_advantage=False,gae_lambda=.92,ent_coef=0.0,max_grad_norm=.8,vf_coef=.4,gamma=.98,v_learning_rate=5e-3, c_learning_rate=1e-2,d_learning_rate=5e-2, use_sde=True,use_rms_prop=False)
#model = SAC("MlpPolicy", env=env, verbose=2, learning_rate=3e-4,buffer_size=

'''
