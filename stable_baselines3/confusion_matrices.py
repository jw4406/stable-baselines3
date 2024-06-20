import torch, torch.autograd as autograd, numpy as np, matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A3C_rarl
from stable_baselines3.a2c.my_pendulum import my_PendulumEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
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
    id="my_half_cheetah",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_HalfCheetahEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=1000,
)

def duel_models(model1, model2, env, num_episodes=10, angle_thresh=20, hold_thresh=100, degrees=True, model_class='pendulum'):
    """
    Simulate matches between two models in the environment.

    Args:
    - model1: First RL model
    - model2: Second RL model
    - env: Custom Gym environment
    - num_episodes: Number of episodes for each match

    Returns:
    - result: A tuple containing the number of wins for model1 and model2
    """
    model1_wins = 0
    model2_wins = 0
    if model_class == "pendulum" or model_class == "pend":
        if degrees is False:
            angle_thresh = angle_thresh * np.pi / 180
        for _ in range(num_episodes):
            obs = env.reset()
            obs_for_env = obs[0]
            obs_for_env = obs_for_env[None, :]
            done = False
            obs_vec = []
            vec_env = model1.get_env()
            obs = vec_env.reset()
            time_up = 0
            #angle_thresh = 10
            while not done:


                action, _, _ = model1.predict(obs, deterministic=True)
                _, dstb_action, _ = model2.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step([[action, dstb_action, 1]])
                vec_env.render()
                #print(reward, action, dstb_action, action + dstb_action, dstb_action)
                x = obs[0,0]
                y = obs[0,1]
                ang = np.arctan2(y,x) * 180 / np.pi
                if np.abs(ang) < angle_thresh:
                    if len(obs_vec) == 0:
                        continue
                    if np.abs(np.arctan2(np.array(obs_vec[-1][0,1]), np.array(obs_vec[-1][0,0])) * 180/np.pi) < angle_thresh:
                        time_up = time_up + 1
                    # VecEnv resets automatically
                    # if done:
                    #   obs = env.reset()
                # Combine actions or decide how to handle multiple actions
                #action = (action1, action2)  # Example; modify based on your env's requirements

                #obs, reward, done, info = env.step(action)
                obs_vec.append(obs)

            if len(obs_vec) < 200:
                # CONTROLLER FAILURE
                # we terminated early because of the swing-failure case
                model2_wins = model2_wins + 1
            elif time_up < hold_thresh:
                # CONTROLLER FAILURE
                # Either:
                # we got up there but didnt get there fast enough
                # or we never got up there.
                # Regardless, this is a controller failure
                model2_wins = model2_wins + 1
            elif time_up >= hold_thresh:
                # CONTROLLER VICTORY
                # We got up there quickly and stayed there!
                model1_wins = model1_wins + 1
    elif model_class == "cheetah" or model_class == "half_cheetah" or model_class == "my_half_cheetah":
        for _ in range(num_episodes):
            rew = 0
            obs = env.reset()
            obs_for_env = obs[0]
            obs_for_env = obs_for_env[None, :]
            done = False
            obs_vec = []
            vec_env = model1.get_env()
            obs = vec_env.reset()
            time_up = 0
            #angle_thresh = 10
            counter = 0
            while not done:
                action, _, _ = model1.predict(obs, deterministic=True)
                _, dstb_action, _ = model2.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step([[action, dstb_action, 1]])
                vec_env.render()
                rew = rew + reward
                if counter == 500:
                    done = True
                else:
                    counter = counter + 1
            if rew > 450:
                model1_wins = model1_wins + 1
            else:
                model2_wins = model2_wins + 1
    return rew, model2_wins


#env = gym.make("my_pendulum", render_mode='human')
model_class = 'cheetah'

if model_class == 'pend':
    env = gym.make("my_pendulum", render_mode='human')
    folder = "/Users/jw4406/Data/Justin/532/dissipativity/stable-baselines3/stable_baselines3/confusion_models/"
    smart_model_path = 'stac_confusion_smart.zip'
    smart = A3C_rarl.load(folder + smart_model_path, env=env)
    smart.spirit = False
    smart_ablation_model_path = 'stac_confusion_ablation.zip'
    smart_ablation = A3C_rarl.load(folder + smart_ablation_model_path, env=env)
    smart_ablation.spirit = False
    baseline_model_path = 'true_pend_baseline_230000_steps.zip'
    baseline = A3C_rarl.load(folder + baseline_model_path, env=env)
    baseline.spirit = False
elif model_class == 'cheetah':
    env = gym.make("my_half_cheetah")
    folder = "/Users/jw4406/Data/Justin/532/dissipativity/stable-baselines3/stable_baselines3/cheetah_model/"
    smart_model_path = 'half_cheetah_della_stac_finished.zip'
    #smart = A3C_rarl.load(folder + smart_model_path, env=env)
    smart = A3C_rarl.load("./half_cheetah_cont_della_6617000_steps.zip", env=env)
    smart.spirit = False
    smart_ablation_model_path = 'stac_confusion_ablation.zip'
    #smart_ablation = A3C_rarl.load(folder + smart_ablation_model_path, env=env)
    smart_ablation = A3C_rarl.load("./half_cheetah_ablation_emergency_2.zip", env=env, fix=True)
    smart_ablation.spirit = False
    #baseline_model_path = '/Users/jw4406/Data/Justin/532/dissipativity/stable-baselines3/stable_baselines3/half_cheetah_baseline_emergency.zip'
    baseline_model_path= "/Users/jw4406/Data/Justin/532/dissipativity/stable-baselines3/stable_baselines3/cheetah_baseline_1860000_steps.zip"
    baseline = A3C_rarl.load(baseline_model_path, env=env)
    baseline.spirit = False

rounds=20 # change later

s_b, s_a, s_s, a_b, a_a, a_s, b_b, b_a, b_s = [], [], [], [], [], [], [], [], []

for i in range(2):
    smartc_baselined_win, baselined_smartc_win = duel_models(smart, baseline, smart.get_env(), num_episodes=rounds, model_class=model_class)
    s_b.append(smartc_baselined_win)
    smartc_ablationd_win, ablationd_smartc_win = duel_models(smart, smart_ablation, smart.get_env(), num_episodes=rounds, model_class=model_class)
    s_a.append(smartc_ablationd_win)
    smartc_smartd_win, smartd_smartc_win = duel_models(smart, smart, smart.get_env(), num_episodes=rounds, model_class=model_class)
    s_s.append(smartc_smartd_win)
    ablationc_baselined_win, baselined_ablationc_win = duel_models(smart_ablation, baseline, smart.get_env(), num_episodes=rounds, model_class=model_class)
    a_b.append(ablationc_baselined_win)
    ablationc_ablationd_win, ablationd_ablationc_win = duel_models(smart_ablation, smart_ablation, smart.get_env(), num_episodes=rounds, model_class=model_class)
    a_a.append(ablationc_ablationd_win)
    ablationc_smartd_win, smartd_ablationc_win = duel_models(smart_ablation, smart, smart.get_env(), num_episodes=rounds, model_class=model_class)
    a_s.append(ablationc_smartd_win)
    baselinec_baselind_win, baselined_baselinec_win = duel_models(baseline, baseline, smart.get_env(), num_episodes=rounds, model_class=model_class)
    b_b.append(baselinec_baselind_win)
    baselinec_ablationd_win, ablationd_baselinec_win = duel_models(baseline, smart_ablation, smart.get_env(), num_episodes=rounds, model_class=model_class)
    b_a.append(baselinec_ablationd_win)
    baselinec_smartd_win, smartd_baselinec_win = duel_models(baseline, smart, smart.get_env(), num_episodes=rounds, model_class=model_class)
    b_s.append(baselinec_smartd_win)

s_b_mean = np.mean(s_b)
s_a_mean = np.mean(s_a)
s_s_mean = np.mean(s_s)
a_b_mean = np.mean(a_b)
a_a_mean = np.mean(a_a)
a_s_mean = np.mean(a_s)
b_b_mean = np.mean(b_b)
b_a_mean = np.mean(b_a)
b_s_mean = np.mean(b_s)

s_b_std = np.std(s_b)
s_a_std = np.std(s_a)
s_s_std = np.std(s_s)
a_b_std = np.std(a_b)
a_a_std = np.std(a_a)
a_s_std = np.std(a_s)
b_b_std = np.std(b_b)
b_a_std = np.std(b_a)
b_s_b_std = np.std(b_s)

1