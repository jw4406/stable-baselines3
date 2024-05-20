import gymnasium as gym
from gymnasium.envs.registration import register
# Example for the CartPole environment
from stable_baselines3.a2c.my_pendulum import my_PendulumEnv
from stable_baselines3.a2c.my_mountain_car_continuous import my_Continuous_MountainCarEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

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
    id="my_mountain_car_continuous",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_Continuous_MountainCarEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=999,
)


import stable_baselines3.a2c
from stable_baselines3 import A2C
from stable_baselines3 import A3C_rarl

#env = gym.make("my_pendulum", render_mode='human')
env = gym.make("my_pendulum")
#env = gym.make("my_pendulum")
#model = A2C("MlpPolicy", env, verbose=1, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,n_steps=8,vf_coef=.4,gamma=.9,learning_rate=1e-4,use_sde=True,use_rms_prop=True)
#model = A2C("MlpPolicy", env=env, verbose=1, normalize_advantage=True, n_steps=100, use_sde=True, use_rms_prop=False)
model = A3C_rarl("MlPAACPolicy", use_stackelberg=True, env=env, verbose=2, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,n_steps=8,vf_coef=.4,gamma=.9,v_learning_rate=1e-4, c_learning_rate=5e-4,d_learning_rate=2e-3, use_sde=True,use_rms_prop=False)
#model = A3C_rarl("MlPAACPolicy", use_stackelberg=True,env=env, verbose=1, normalize_advantage=True, n_steps=8, v_learning_rate=1e-4, c_learning_rate=5e-3,d_learning_rate=1e-2, use_sde=True, use_rms_prop=False)

#model = A3C_rarl.load("didnt_break_anything_pend.zip", env=env)
#model = A2C.load("mcc_0.zip", env=env)
#model = A3C_rarl.load("adv_pendulum_split_0.zip", env=env)
#model.v_learning_rate = 1e-6
#model.c_learning_rate = 1e-7
#model.d_learning_rate = 6e-7
#model.n_steps = 7
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
model.learn(total_timesteps=5_000_000, callback=eval_callback)
#callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
#eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
#model.save("didnt_break_anything_pend.zip")
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
