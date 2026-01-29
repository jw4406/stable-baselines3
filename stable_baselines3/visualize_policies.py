#from stable_baselines3.common.clean_new_policies import CleanNewPolicies
from stable_baselines3.main.common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from stable_baselines3.a2c.my_ant_v5 import my_AntEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from gymnasium import register
import gymnasium as gym
import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
import imageio
model_path = "/home/jw4406/codebase/stable-baselines3/stable_baselines3/main/trained_models/tasks/todo/my_ant_ego_1.0_adv_0.5_11000000_steps.task"

register(
    id="my_ant",
    entry_point=my_AntEnv,
    max_episode_steps=1000,
)

env = gym.make("my_ant", render_mode='rgb_array')
#env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix="my_ant")
# vec_env = DummyVecEnv([lambda: gym.make("my_ant", render_mode="rgb_array")])
model = CleanDerivativeFreeSPAR.load(model_path, env=env, num_perturbed=1)
done = False
images = []
obs = model.env.reset()
#env.start_video_recorder()
img = model.env.render()
cum_rew = 0
while not done:
    images.append(img)
    with th.no_grad():
        ego_actions, ego_log_probs, adv_actions, adv_log_probs, values, q_values = model.policy(obs_as_tensor(obs, model.device), deterministic=False, ego_forward=True, adv_forward=True, zero_ego_action=False, zero_adv_action=True)
    actions = ego_actions.cpu().numpy()
    actions_other = adv_actions.cpu().numpy()
    clipped_actions = np.hstack([actions, actions_other])
    obs, reward, done, info = model.env.step(clipped_actions)
    img = model.env.render()
    cum_rew += reward
print(f"Cumulative reward: {cum_rew}")
model.env.close()

imageio.mimsave("my_ant.gif", [np.array(img) for img in images], fps=29)
#model.predict(env.reset())