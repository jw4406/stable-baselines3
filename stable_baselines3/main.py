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
    max_episode_steps=500,
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
from stable_baselines3 import SAC


#env = gym.make("MountainCarContinuous-v0")
#env = gym.make("my_half_cheetah", render_mode='human')
env = gym.make("my_pendulum", render_mode='human')

#model = A3C_rarl("MlPAACPolicy", use_stackelberg=False, env=env, verbose=2, n_steps=8, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,vf_coef=.4,gamma=.9,v_learning_rate=5e-4, c_learning_rate=5e-4,d_learning_rate=5e-4, use_sde=True,use_rms_prop=False, device='cpu')



#model = A3C_rarl("MlPAACPolicy", use_stackelberg=False, env=env, verbose=2, n_steps=8, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,vf_coef=.4,gamma=.9,v_learning_rate=linear_schedule(5e-4), c_learning_rate=linear_schedule(5e-4),d_learning_rate=linear_schedule(5e-4), use_sde=True,use_rms_prop=False, device='auto')
#model = A3C_rarl("MlPAACPolicy", use_stackelberg=True, env=env, verbose=2, n_steps=8, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,vf_coef=.4,gamma=.9,v_learning_rate=linear_schedule(5e-4), c_learning_rate=linear_schedule(1e-3),d_learning_rate=linear_schedule(5e-3), use_sde=True,use_rms_prop=False, device='auto')

#model = A3C_rarl("MlPAACPolicy", use_stackelberg=True, env=env, verbose=2, n_steps=100, normalize_advantage=False,v_learning_rate=linear_schedule(1e-5), c_learning_rate=linear_schedule(5e-5),d_learning_rate=linear_schedule(1e-4), use_sde=True,use_rms_prop=False, device='auto')

model = A3C_rarl("MlPAACPolicy", use_stackelberg=True, env=env, verbose=2, n_steps=8, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,vf_coef=.4,gamma=.9,v_learning_rate=linear_schedule(1e-3), c_learning_rate=linear_schedule(2e-3),d_learning_rate=linear_schedule(1e-2), use_sde=True,use_rms_prop=False, device='auto', seed=42069)


#model = A3C_rarl("MlPAACPolicy", use_stackelberg=True, env=env, verbose=2, n_steps=100, normalize_advantage=False,v_learning_rate=linear_schedule(5e-4), c_learning_rate=linear_schedule(1e-3),d_learning_rate=linear_schedule(5e-3), use_sde=True,use_rms_prop=False, device='auto')
#model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.4, .4, (2,), dtype=np.float32), use_stackelberg=False,
#                     env=env, verbose=2, n_steps=32, normalize_advantage=False, gae_lambda=.95, ent_coef=0.0,
#                     max_grad_norm=.7, vf_coef=.4, gamma=.95, v_learning_rate=linear_schedule(5e-4),
#                     c_learning_rate=linear_schedule(5e-3), d_learning_rate=linear_schedule(9e-3), use_sde=True,
#                     use_rms_prop=False)
#model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.4, .4, (2,), dtype=np.float32), use_stackelberg=False,
                     #env=env, verbose=2, n_steps=32, normalize_advantage=False, gae_lambda=.95, ent_coef=0.0,
                     #max_grad_norm=.7, vf_coef=.4, gamma=.95, v_learning_rate=linear_schedule(5e-4),
                     #c_learning_rate=linear_schedule(5e-4), d_learning_rate=linear_schedule(5e-4), use_sde=True,
                     #use_rms_prop=False)
#model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (2,), dtype=np.float32), use_stackelberg=True, env=env, verbose=2, n_steps=512, normalize_advantage=False,gae_lambda=.92,ent_coef=0.0,max_grad_norm=.8,vf_coef=.4,gamma=.98,v_learning_rate=5e-4, c_learning_rate=1e-3,d_learning_rate=5e-3, use_sde=True,use_rms_prop=False)
#model=A2C("MlpPolicy", normalize_advantage=True, verbose=2, env=env, n_steps=100, learning_rate=linear_schedule(3e-4), use_sde=True, use_rms_prop=False)
#model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (2,), dtype=np.float32), use_stackelberg=True, env=env, verbose=2, n_steps=512, normalize_advantage=False,gae_lambda=.92,ent_coef=0.0,max_grad_norm=.8,vf_coef=.4,gamma=.98,v_learning_rate=5e-3, c_learning_rate=1e-2,d_learning_rate=5e-2, use_sde=True,use_rms_prop=False)
#model = SAC("MlpPolicy", env=env, verbose=2, learning_rate=3e-4,buffer_size=50000, batch_size=512, ent_coef=0.1, train_freq=32, gradient_steps=32, gamma=0.9999, tau=0.01, use_sde=True)

#model = SMART("MlPAACPolicy", use_stackelberg=True, dstb_action_space=Box(-.3, .3, (1,), dtype=np.float32), learning_starts=100, env=env, verbose=2, v_learning_rate=linear_schedule(5e-4), c_learning_rate=linear_schedule(7e-4), d_learning_rate=linear_schedule(5e-3),buffer_size=25000, batch_size=128, train_freq=32, gradient_steps=32, gamma=0.99, tau=0.01, use_sde=True)
#model=A2C("MlpPolicy", normalize_advantage=True, verbose=2, env=env, n_steps=100, learning_rate=linear_schedule(3e-4), use_sde=True, use_rms_prop=False)
#model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (2,), dtype=np.float32), use_stackelberg=False, env=env, verbose=2, n_steps=32, normalize_advantage=False,gae_lambda=.95,ent_coef=0.0,max_grad_norm=.7,vf_coef=.4,gamma=.95,v_learning_rate=linear_schedule(5e-5), c_learning_rate=linear_schedule(5e-5),d_learning_rate=linear_schedule(5e-5), use_sde=True,use_rms_prop=False)
#model = SAC("MlpPolicy", env=env, verbose=2, learning_rate=3e-4,buffer_size=50000, batch_size=512, ent_coef=0.1, train_freq=32, gradient_steps=32, gamma=0.9999, tau=0.01, use_sde=True)

#model = SMART("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (1,), dtype=np.float32), learning_starts=0, env=env, verbose=2, v_learning_rate=5e-4, c_learning_rate=1e-3, d_learning_rate=5e-3,buffer_size=50000, batch_size=256, train_freq=32, gradient_steps=64, gamma=0.9999, tau=0.01, use_sde=True, device='cpu')
#model = A3C_rarl("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (2,), dtype=np.float32), use_stackelberg=False, env=env, verbose=2, n_steps=32, normalize_advantage=False,gae_lambda=.95,ent_coef=0.0,max_grad_norm=.7,vf_coef=.4,gamma=.95,v_learning_rate=linear_schedule(5e-4), c_learning_rate=linear_schedule(1e-3),d_learning_rate=linear_schedule(5e-3), use_sde=True,use_rms_prop=False)
#model = SMART("MlPAACPolicy", dstb_action_space=Box(-.3, .3, (1,), dtype=np.float32), learning_starts=0, env=env, verbose=2, v_learning_rate=5e-4, c_learning_rate=1e-3, d_learning_rate=5e-3,buffer_size=25000, batch_size=128, train_freq=32, gradient_steps=32, gamma=0.9999, tau=0.01, use_sde=True)


#model = A3C_rarl("MlPAACPolicy", use_stackelberg=False,env=env, verbose=1, normalize_advantage=True, n_steps=100, v_learning_rate=5e-4, c_learning_rate=1e-3,d_learning_rate=5e-3, use_sde=True, use_rms_prop=False)


#model = A3C_rarl.load("./models/pend_smart_388000_steps.zip", env=env)
#model = A3C_rarl.load("./half_cheetah_", env=env)
#model = A3C_rarl("MlPAACPolicy", use_stackelberg=False,env=env, verbose=2, normalize_advantage=False, n_steps=100,v_learning_rate=linear_schedule(3e-4), c_learning_rate=linear_schedule(6e-4),d_learning_rate=linear_schedule(1.2e-3), use_sde=True, use_rms_prop=False)
#model = A3C_rarl.load("./cheetah_model/half_cheetah_della_stac_finished.zip", env=env)
#model = A3C_rarl.load("./half_cheetah_baseline/half_cheetah_baseline.zip", env=env)
#model = A3C_rarl.load("./half_cheetah_ablation/stac_half_cheetah_smart_ablation_2268000_steps.zip", env=env)
#model = A3C_rarl.load("./confusion_models/stac_pend_model_vis_1000000_steps.zip", env=env)
#model = A3C_rarl.load("./confusion_models/stac_confusion_ablation.zip", env=env)
#model = A3C_rarl.load("./stac_tau_sweep_cheetah_rew_1500_take1_5.000000_1049000_steps.zip", env=env)
#model = A3C_rarl.load("./logs/stac_heavy_280000_steps.zip", env=env)
#model = A3C_rarl.load("./logs/stac_pend_heavy_1_863000_steps.zip", env=env)
#model = A3C_rarl.load("./test_2.zip", env=env)
#model = A3C_rarl.load("./stac_pend_heavy_3.zip", env=env)
#model = A3C_rarl.load("./stac_pend_heavy_3_cont_991000_steps.zip", env=env)
#model = A3C_rarl.load("./baseline_test_1.zip", env=env)
model = A3C_rarl.load("./logs/1616_8and2_stac_2142000_steps.zip", env=env)
model.spirit=False
#model = SMART.load("./sac_pend_t2_330000_steps.zip", env=env)
#model = A3C_rarl.load("./pend_smart_410400_steps.zip", env=env)
#model = A3C_rarl.load("./stac_pend_training_curve.zip", env=env)
#A3C_rarl.load("./mcc_sac_test_newres")
#model = A3C_rarl.load("./mcc_sac_test_newrew_159100_steps.zip", env=env)
#model = A3C_rarl.load("./mcc_sac_test_newrew_300000_steps.zip", env=env)
#model = A2C.load("mcc_0.zip", env=env)#
#model = A3C_rarl.load("./conf_stac_pend_ft.zip", env=env)

#model = A2C.load("mcc_0.zip", env=env)
#model = A3C_rarl.load("adv_pendulum_split_0.zip", env=env)
#model.v_learning_rate = linear_schedule(1e-4)
#model.c_learning_rate = linear_schedule(5e-4)
#model.d_learning_rate = linear_schedule(1e-3)
model.lr_schedule = [linear_schedule(1e-4), linear_schedule(5e-4), linear_schedule(1e-3)]
model.policy.value_optimizer.param_groups[0]['lr'] =1e-4
model.policy.ctrl_optimizer.param_groups[0]['lr'] = 5e-4
model.policy.dstb_optimizer.param_groups[0]['lr'] = 1e-3
#model.n_steps = 7
#model.use_stackelberg=True




callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-150, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1, n_eval_episodes=10, jobid=args.jobid)

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs/",
  name_prefix='1616_8and2_stac',
)

callback_list = CallbackList([eval_callback, checkpoint_callback])

#model.learn(total_timesteps=7_500_000, callback=callback_list)

#callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
#eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
#model.save("stac_weightdecay_5e_ud_28_split.zip")
#model.learn(total_timesteps=5000000, callback=callback_list)

#callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
#eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
#model.save("1616_6and4_baseline.zip")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100000):
    action, dstb_action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step([[action, dstb_action, i]])
    #print(reward, action, dstb_action, action + dstb_action, dstb_action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
