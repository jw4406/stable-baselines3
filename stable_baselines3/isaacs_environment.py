import os
import sys
import copy
import numpy as np
import argparse
from functools import partial
from shutil import copyfile
from omegaconf import OmegaConf
from simulators import Go2PybulletZeroSumEnv
from simulators import PrintLogger, save_obj
from stable_baselines3 import A3C_rarl
import gymnasium.spaces
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList
def main(config_file):
  cfg = OmegaConf.load(config_file)
  os.makedirs(cfg.solver.out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(cfg.solver.out_folder, 'config.yaml'))
  log_path = os.path.join(cfg.solver.out_folder, 'log.txt')
  if os.path.exists(log_path):
    os.remove(log_path)
  sys.stdout = PrintLogger(log_path)
  sys.stderr = PrintLogger(log_path)

  env_class = Go2PybulletZeroSumEnv
  cfg.cost = None
  env = env_class(cfg.environment, cfg.agent, cfg.cost)
  env.step_keep_constraints = False
  env.report()
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
  dstb_action_space = gymnasium.spaces.Box(low=np.array([-1., -1., -1., -.015, -.04, -.05]), high=np.array([1., 1., 1., .15, .04, .05]), shape=(6,), dtype=np.float32)
  model = A3C_rarl("MlPAACPolicy", dstb_action_space=dstb_action_space, use_stackelberg=False, env=env, verbose=2, n_steps=20, normalize_advantage=False,
                   gae_lambda=.9, ent_coef=0.0, max_grad_norm=.5, vf_coef=.4, gamma=.9,
                   v_learning_rate=linear_schedule(5e-4), c_learning_rate=linear_schedule(1e-4),
                   d_learning_rate=linear_schedule(5e-4), use_sde=True, use_rms_prop=False, device='auto', spirit=True)
  callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
  eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1, n_eval_episodes=10,
                               jobid=None)
  checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./",
    name_prefix='spirit',
  )

  callback_list = CallbackList([eval_callback, checkpoint_callback])

  model.learn(total_timesteps=5_000_000)  # , callback=callback_list)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str, default=os.path.join("config", "isaacs.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
