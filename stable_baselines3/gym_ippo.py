#import gym
import gymnasium
from gymnasium.envs.registration import register
from main.common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
import wandb
import os
import argparse
from stable_baselines3.a2c.my_pendulum import my_PendulumEnv
from stable_baselines3.a2c.my_walker2d_v4 import my_Walker2dEnv
from stable_baselines3.a2c.my_mountain_car_continuous import my_Continuous_MountainCarEnv
from stable_baselines3.a2c.my_half_cheetah import my_HalfCheetahEnv
from stable_baselines3.a2c.my_hopper_v5 import my_HopperEnv
from stable_baselines3.a2c.my_ant_v5 import my_AntEnv
# from stable_baselines3.common.adversarial_envs.my_pendulum import my_PendulumEnv
# from stable_baselines3.common.adversarial_envs.my_walker2d_v4 import my_Walker2dEnv
# from stable_baselines3.common.adversarial_envs.my_mountain_car_continuous import my_Continuous_MountainCarEnv
# from stable_baselines3.common.adversarial_envs.my_half_cheetah import my_HalfCheetahEnv
# from stable_baselines3.common.adversarial_envs.my_hopper_v5 import my_HopperEnv
# from stable_baselines3.common.adversarial_envs.my_ant_v5 import my_AntEnv
from stable_baselines3.common.callbacks import SACheckpointCallback, FileQueueTriggerCallback, CallbackList
def critic_decay_schedule(initial_value):
    return lambda progress: initial_value * (1 - progress)
def actor_decay_schedule(initial_value):
    return lambda progress: initial_value * (1 - progress)
register(
    # unique identifier for the env `name-version`
    id="my_pendulum",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_PendulumEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=200,
)

def env_generator(STATE=None, ego_strength=1.5, adv_strength=0.5):
    env_name = STATE[0].split(".")[1]
    return gymnasium.make(env_name, ego_strength=ego_strength, adv_strength=adv_strength)
PLAYER = "ego0"
OPPONENT_LIST = ["adv0"]
TOTAL_TIMESTEPS = 100000000
current_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(current_dir, "trained_models/main_checkpoint_models")
TASK_DIR = os.path.join(current_dir, "trained_models/tasks")
BR_CHECKPOINT_DIR = os.path.join(current_dir, "trained_models/br_checkpoint_models")
def main(args):
    env_name = args.env_name
    model_name_prefix = f"{env_name}_ego_{args.ego_strength}_adv_{args.adv_strength}"
    print("CURRENT MODEL NAME PREFIX: %s" % model_name_prefix)
    ego_strength = args.ego_strength
    adv_strength = args.adv_strength

    STATE = ["Champion.%s.%sVs%s.2Player.state" % (env_name, PLAYER, OPPONENT_LIST[0])]
    env = env_generator(STATE=STATE, ego_strength=ego_strength, adv_strength=adv_strength)
    state_list = STATE
    finetune_model = CleanDerivativeFreeSPAR(
            policy="AACCnnPolicy",
            env=env,
            device="cuda",
            c_learning_rate=args.c_lr,
            d_learning_rate=args.d_lr,
            v_learning_rate=args.v_lr,
            verbose=2,
            n_steps=args.num_env_steps,
            batch_size=300,
            n_epochs=4,
            state_list=state_list,
            envs_per_matchup=1,
            env_generator_func=env_generator,
            num_adversaries=1,
            n_env_per_adv=1,
            seed= 0,
            target_kl=None,
            use_mirror=False,
            use_lr_annealing=args.use_lr_annealing,
            lr_anneal_coeff=args.lr_anneal_coeff,
            ego_strength=ego_strength,
            adv_strength=adv_strength
        )
    checkpoint_interval = args.checkpoint_interval
    checkpoint_callback = SACheckpointCallback(save_freq=checkpoint_interval, save_path=args.save_dir,
                                               name_prefix=f"{model_name_prefix}")
    file_queue_callback = FileQueueTriggerCallback(
        task_dir=TASK_DIR,
        use_mirror=False,
        num_workers=2,
        save_freq=checkpoint_interval,
        save_path=BR_CHECKPOINT_DIR,
        name_prefix=f"{model_name_prefix}"
    )
    callback_list = CallbackList([checkpoint_callback, file_queue_callback])
    finetune_model.learn(update_adversary=True,total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, choices=["my_pendulum", "my_walker2d", "my_mountain_car", "my_half_cheetah", "my_hopper", "my_ant"], required=True, default="my_pendulum")
    parser.add_argument("--c_lr", type=float, required=True, default=1e-6)
    parser.add_argument("--d_lr", type=float, required=True, default=2e-6)
    parser.add_argument("--v_lr", type=float, required=True, default=4e-6)
    parser.add_argument("--num_perturbs", type=int, required=True, default=10)
    parser.add_argument("--load_path", type=str, required=True, default=None)
    parser.add_argument("--continue_training", type=bool, required=True, default=False)
    parser.add_argument("--model-file", type=str, required=True, default=None)
    parser.add_argument("--use_lr_annealing", type=bool, required=True, default=False)
    parser.add_argument("--lr_anneal_coeff", type=float, required=True, default=0.995)
    parser.add_argument("--checkpoint_interval", type=int, required=True, default=100000)
    parser.add_argument("--num_env_steps", type=int, required=True, default=1024)
    parser.add_argument("--envs_per_matchup", type=int, required=True, default=1)
    parser.add_argument("--ego_strength", type=float, required=True, default=1.5)
    parser.add_argument("--adv_strength", type=float, required=True, default=0.5)
    parser.add_argument("--save_dir", type=str, required=True, default=CHECKPOINT_DIR)
    args = parser.parse_args()
    wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    wandb.init(project="gym_ippo",
               entity='jw4406',
               config={"c_lr": args.c_lr,
                       "d_lr": args.d_lr,
                       "v_lr": args.v_lr,
                       "num_env_steps": args.num_env_steps,
                       "envs_per_matchup": args.envs_per_matchup})
    main(args)