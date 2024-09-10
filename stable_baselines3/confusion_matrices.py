import torch, torch.autograd as autograd, numpy as np, matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A3C_rarl, SMART
from stable_baselines3.a2c.my_pendulum import my_PendulumEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
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
    id="my_half_cheetah",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_HalfCheetahEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=1000,
)


def duel_models(model1, model2, env, num_episodes=10, angle_thresh=20, hold_thresh=150, degrees=True, model_class='pendulum', sd=False):

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
        controller_wins = []
        rew_list = []
        for k in range(len(model1)):
            for l in range(len(model2)):
                model1_wins = 0
                model2_wins = 0
                my_env = model1[k].get_env()
                for _ in range(num_episodes):
                    #obs = env.reset()
                    rew_test = 0
                    #obs_for_env = obs[0]
                    #obs_for_env = obs_for_env[None, :]
                    done = False
                    obs_vec = []
                    obs = my_env.reset()
                    time_up = 0
                    #angle_thresh = 10
                    while not done:


                        action, _, _ = model1[k].predict(obs, deterministic=True)
                        if sd is True:
                            _, dstb_action, _ = model2[l].policy.policy_memory[l].predict(obs, deterministic=True)
                        else:
                            _, dstb_action, _ = model2[l].predict(obs, deterministic=True)
                        obs, reward, done, info = my_env.step([[action, dstb_action, 1]])
                        rew_test = rew_test + reward
                        #vec_env.render()
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

                    if len(obs_vec) < 500:
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
                controller_wins.append(model1_wins)
                rew_list.append(rew_test)
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
                rew_test = rew_test + reward
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
    return controller_wins, rew_list


#env = gym.make("my_pendulum", render_mode='human')
model_class = 'pend'

if model_class == 'pend':
    nums = np.arange(10)
    nums = [0,1,3,4]
    #nums = [0,4,6,7]
    #nums = [1,2,3,5,6,7,8]
    seeds = [3721, 9323764, 834981, 9274, 42069, 92048, 109475, 373095, 5, 92038]
    #nums = [0,1]
    env = gym.make("my_pendulum")
    folder = "/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/"
    smart_model_list = []
    ablation_model_list = []
    baseline_model_list = []
    # folder = "/home/jw4406/codebase/stable-baselines3/stable_baselines3/logs/"
    #smart_model_path = 'stac_train_pend_parallel_FINISHED_%d.zip' % nums[i]
    #smart_model_path = 'leaderboard_10_trained_d_no_grad_tss_zoo_ud_55_3.zip'
    for i in range(len(nums)):

        #smart_model_path = 'leaderboard_10_trained_d_no_grad_tss_zoo_ud_46.zip'
        #smart_model_path = "stac_fulltrain_pend_adversarial_populations_10_FINISHED_ud_55_%d.zip" % nums[i]
        #smart_model_path = 'stac_fulltrain_pend_adversarial_populations_10_FINISHED_ud_46_%d_cont2.zip' % nums[i]
        #smart_model_path = "adv_pop_ws10cont2_exp_decay_lr25_ud46_%d_777000_steps.zip" % nums[i]
        #smart_model_path = "adv_pop_ws10cont1_exp_decay_lr25_ud46_%d_429000_steps.zip" % nums[i]
        #smart_model_path = 'stac_complete_new_pretrain_EXP_DECAY_FROM_750000_COMPLETE_advpop10_%d.zip' % nums[i]
        #smart_model_path = 'stac_completely_new_pretrain_exp_2_5mil_advpop_10_%d_767000_steps.zip' % nums[i]
        #smart_model_path = 'iso/stac_completely_new_pretrain_linear_phase_5mil_tss_25_advpop_10_%d_2400000_steps.zip' % nums[i]
        #smart_model_path = 'stac_completely_new_pretrain_linear_5mil_advpop_10_%d_1120000_steps.zip' % nums[i]
        env.reset(seed=seeds[0])
        #smart = A3C_rarl.load(folder + smart_model_path, env=env)
        smart = A3C_rarl.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/smart_trained_%d.zip" % nums[i], env=env)
        smart = SMART.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/stsac_baseline_pend_hl3_len200_28_ef_%d_90000_steps.zip" % nums[i], env=env, device='cuda')
        #smart = A3C_rarl.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/pend_stac_cont_%d_73000_steps.zip" % nums[i], env=env)
        #smart = A3C_rarl.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/stac_55_corrected/stac_pend_len200_ud_55_%d_267000_steps.zip" % nums[i], env=env)
        #try:
        #    smart = A3C_rarl.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/test_2_%d_150000_steps.zip" % nums[i], env=env)
        #except:
        #    continue
        #smart.save("smart_trained_%d.zip" % nums[i])
        #pretrain_path = "/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/"
        #pretrain_model_name = "stac_pretrain_pend_parallel_FINISHED_ud_46_%d.zip" % i
        #smart.policy.policy_memory[i] = A3C_rarl.load(pretrain_path + pretrain_model_name, env=env).policy
        smart.spirit = False
        smart_model_list.append(smart)
    for i in range(len(nums)):
        '''#folder = "/home/jw4406/codebase/stable-baselines3/stable_baselines3/logs/"
        smart_model_path = 'stac_train_pend_parallel_FINISHED_%d.zip' % nums[i]
        env.reset(seed=seeds[i])
        smart = A3C_rarl.load(folder + smart_model_path, env=env)
        smart.spirit = False
        smart_model_list.append(smart)
        '''
        folder = "/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/"
        #smart_ablation_model_path = 'ablation_train_pend_parallel_FINISHED_%d.zip' % nums[i]
        #smart_ablation = A3C_rarl.load(folder + smart_ablation_model_path, env=env)
        #smart_ablation.spirit = False
        #ablation_model_list.append(smart_ablation)
        baseline_model_path = 'stac_train_pend_parallel_FINISHED_wd_53_ud_55_%d.zip' % nums[i]
        #baseline_model_path = 'baseline_pretrain_parallel_pend_tss_zoo_ud_55_%d_553000_steps.zip' % nums[i]
        #baseline_model_path = 'adversarial_populations_pretrain_stac_size_10_ud_46_iter_%d_1301000_steps.zip' % nums[i]
        #baseline_model_path = 'stac_pretrain_pend_adversarial_populations_10_FINISHED_ud_46_%d.zip' % nums[i]
        #baseline_model_path = 'baseline_adv_pop_ws10_exp_decay_ud46_%d_1348000_steps.zip' % nums[i]
        #baseline_model_path = 'stac_complete_new_pretrain_EXP_DECAY_FROM_750000_COMPLETE_advpop10_%d.zip' % nums[i]
        #baseline_model_path = 'baseline_adv_pop_ws10_exp_decay_ud46_%d_500000_steps.zip' % nums[i]
        #baseline = A3C_rarl.load(folder + baseline_model_path, env=env)
        baseline = A3C_rarl.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/baseline_trained_%d.zip" % nums[i], env=env)
        #baseline = SMART.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/stsac_models/stsac_baseline_pend_hl3_len200_64_%d_73000_steps.zip" % nums[i], env=env, device='cpu')
        baseline = SMART.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/stsac_baseline_pend_hl3_len200_28_ef_%d_90000_steps.zip" % nums[i], env=env, device='cuda')
        #baseline = A3C_rarl.load(
        #    "/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/pend_baseline_cont_%d_73000_steps.zip" % nums[i], env=env)
        baseline.spirit = False
        baseline_model_list.append(baseline)
        #baseline.save("baseline_trained_%d.zip" % nums[i])
    for i in range(len(nums)):
        folder = "/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/"
        #ablation_model_path = 'adversarial_populations_pretrain_ablation_size_10_ud_46_iter_%d_1500000_steps.zip' % nums[i]
        ablation_model_path = 'ablation_search_iso/ablation_linear_explore_rejection_%d_268000_steps.zip' % i
        #ablation_model_path = 'ablation_slow_critic_linear_%d_1000000_steps.zip' % nums[i]
        #ablation_model_path = 'ablation_exp_decay_cont1_%d_38000_steps.zip' % nums[i]
        #ablation_model_path = 'adversarial_populations_pretrain_ablation_size_10_ud_46_larger_tss_iter_%d_750000_steps.zip' % nums[i]
        #ablation_model_path = 'ablation_completely_new_pretrain_linear_advpop_10_%d_1800000_steps.zip' % nums[i]
        #ablation_model_path = 'ablation_completely_new_pretrain_exp_decay_tautau1010_25_advpop_10_%d_191000_steps.zip' % nums[i]
        #try:
        #    ablation = A3C_rarl.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/ablation_search_iso/lr_twostep_pend_ablation_16166464_%d_961000_steps.zip" % i, env=env)
        #except:
        #    continue
        #try:
        #    ablation = A3C_rarl.load(folder + ablation_model_path, env=env, device='cpu')
        #except:
        #    continue
        #ablation.seed = baseline_model_list[i].seed
        ablation = SMART.load("/home/jw4406/codebase/stable-baselines3/stable_baselines3/competitive_models/stsac_ablation_pend_hl3_len200_28_ef_%d_90000_steps.zip" % nums[i], env=env, device='cuda')
        ablation.spirit = False
        ablation_model_list.append(ablation)
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

rounds=100 # change later

s_b, s_a, s_s, a_b, a_a, a_s, b_b, b_a, b_s = [], [], [], [], [], [], [], [], []

for i in range(1):
    smartc_baselined_win, baselined_smartc_win = duel_models(smart_model_list, baseline_model_list, smart.get_env(),
                                                             num_episodes=rounds, model_class=model_class)
    smartc_smartd_win, smartd_smartc_win = duel_models(smart_model_list, smart_model_list, smart.get_env(),
                                                       num_episodes=rounds, model_class=model_class, sd=False) #!!

    #s_b.append(smartc_baselined_win)

    #TEST
    baselinec_smartd_win, smartd_baselinec_win = duel_models(baseline_model_list, smart_model_list, smart.get_env(),
                                                             num_episodes=rounds, model_class=model_class, sd=False) #!!
    #b_s.append(baselinec_smartd_win)
    #smartc_smartd_win, smartd_smartc_win = duel_models(smart_model_list, smart_model_list, smart.get_env(),
    #                                                   num_episodes=rounds, model_class=model_class, sd=True)
    #s_s.append(smartc_smartd_win)

    baselinec_baselind_win, baselined_baselinec_win = duel_models(baseline_model_list, baseline_model_list,
                                                                  smart.get_env(), num_episodes=rounds,
                                                                  model_class=model_class)
    #b_b.append(baselinec_baselind_win)

    smartc_ablationd_win, ablationd_smartc_win = duel_models(smart_model_list, ablation_model_list, smart.get_env(), num_episodes=rounds, model_class=model_class)
    s_a.append(smartc_ablationd_win)
    #smartc_smartd_win, smartd_smartc_win = duel_models(smart_model_list, smart_model_list, smart.get_env(), num_episodes=rounds, model_class=model_class)
    #s_s.append(smartc_smartd_win)
    ablationc_baselined_win, baselined_ablationc_win = duel_models(ablation_model_list, baseline_model_list, smart.get_env(), num_episodes=rounds, model_class=model_class)
    #a_b.append(ablationc_baselined_win)
    ablationc_ablationd_win, ablationd_ablationc_win = duel_models(ablation_model_list, ablation_model_list, smart.get_env(), num_episodes=rounds, model_class=model_class)
    #a_a.append(ablationc_ablationd_win)
    ablationc_smartd_win, smartd_ablationc_win = duel_models(ablation_model_list, smart_model_list, smart.get_env(), num_episodes=rounds, model_class=model_class, sd=False)
    #a_s.append(ablationc_smartd_win)
    #baselinec_baselind_win, baselined_baselinec_win = duel_models(baseline_model_list, baseline_model_list, smart.get_env(), num_episodes=rounds, model_class=model_class)
    #b_b.append(baselinec_baselind_win)
    baselinec_ablationd_win, ablationd_baselinec_win = duel_models(baseline_model_list, ablation_model_list, smart.get_env(), num_episodes=rounds, model_class=model_class)
    b_a.append(baselinec_ablationd_win)
    #baselinec_smartd_win, smartd_baselinec_win = duel_models(baseline_model_list, smart_model_list, smart.get_env(), num_episodes=rounds, model_class=model_class)
    #b_s.append(baselinec_smartd_win)

s_b_mean = np.mean(s_b) * 20
s_a_mean = np.mean(s_a) * 20
s_s_mean = np.mean(s_s) * 20
a_b_mean = np.mean(a_b) * 20
a_a_mean = np.mean(a_a) * 20
a_s_mean = np.mean(a_s) * 20
b_b_mean = np.mean(b_b) * 20
b_a_mean = np.mean(b_a) * 20
b_s_mean = np.mean(b_s) * 20

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