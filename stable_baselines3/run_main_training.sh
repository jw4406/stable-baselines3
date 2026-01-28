#! /bin/bash

# options: my_pendulum, my_half_cheetah, my_hopper, my_ant
ENV_NAME=my_half_cheetah

RUN_LIVE=False

c_lr=1e-4
d_lr=2e-4
v_lr=4e-4

load_path=""
continue_training=False
model_file=""
use_lr_annealing=False
lr_anneal_coeff=.995
ckpt_interval=100000
n_env_steps=4096

ego_strength=1.5
adv_strength=.5

save_dir=/home/jw4406/codebase/stable-baselines3/stable_baselines3/main/trained_models/tasks/main_checkpoint_models/
CMD=(
  python gym_ippo.py
  --env_name "$ENV_NAME"
  --c_lr "$c_lr"
  --d_lr "$d_lr"
  --v_lr "$v_lr"
  --num_perturbs 10
  --load_path "$load_path"
  --continue_training "$continue_training"
  --model-file "$model_file"
  --use_lr_annealing "$use_lr_annealing"
  --lr_anneal_coeff "$lr_anneal_coeff"
  --checkpoint_interval "$ckpt_interval"
  --num_env_steps "$n_env_steps"
  --envs_per_matchup 1
  --ego_strength "$ego_strength"
  --adv_strength "$adv_strength"
  --save_dir "$save_dir"
)

if [[ "$RUN_LIVE" == "True" ]]; then
  "${CMD[@]}"
else
  nohup "${CMD[@]}" \
    > "${ENV_NAME}_sanity_test_zero_adv.out" 2>&1 &
fi

