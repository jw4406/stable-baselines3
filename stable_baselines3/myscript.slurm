#!/bin/bash
#SBATCH --job-name=RARL_LSE         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --partition=gpu-ee
#SBATCH --mail-user=jw4406@princeton.edu
echo $CUDA_VISIBLE_DEVICES
JOBID=$SLURM_JOB_ID


cd $WORKDIR
mkdir -p $JOBID
cd $JOBID
cp -r $HOME/stable-baselines3 ./
cd stable-baselines3/stable_baselines3/

module purge
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda3/2024.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/2024.02/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate mujoco_sb3
##python run_experiments.py $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
#python main.py --jobid $JOBID
python tau_sweep.py
