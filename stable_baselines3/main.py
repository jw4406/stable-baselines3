import gymnasium as gym
from gymnasium.envs.registration import register
# Example for the CartPole environment
from stable_baselines3.a2c.my_pendulum import my_PendulumEnv
register(
    # unique identifier for the env `name-version`
    id="my_pendulum",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_PendulumEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=200,
)


import stable_baselines3.a2c
from stable_baselines3 import A2C
from stable_baselines3 import A3C_rarl

env = gym.make("my_pendulum", render_mode='human')
#model = A2C("MlpPolicy", env, verbose=1, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,n_steps=8,vf_coef=.4,gamma=.9,learning_rate=1e-4,use_sde=True,use_rms_prop=True)
model = A3C_rarl("MlPAACPolicy", env='my_pendulum', verbose=2, normalize_advantage=False,gae_lambda=.9,ent_coef=0.0,max_grad_norm=.5,n_steps=200,vf_coef=.4,gamma=.9,v_learning_rate=1e-4, c_learning_rate=1e-3,d_learning_rate=5e-3, use_sde=True,use_rms_prop=False)
model = A3C_rarl.load("a2c_pendulum_take_8.zip", env=env)
#model.learn(total_timesteps=5_000_000)
#model.save("a2c_pendulum_take_8")
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, dstb_actions, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step([[action, dstb_actions, i]])
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()