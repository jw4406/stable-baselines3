#!/bin/bash

# Number of parallel br_worker instances to run
NUM_WORKERS=2

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to br_worker.py
BR_WORKER_PATH="${SCRIPT_DIR}/stable_baselines3/br_worker.py"

# Arguments from launch.json
NUM_BRS="2"
DEBUG="False"
EVAL_ONLY="False"
PROJ_NAME="pendulum_br_training"
ANALYSIS_UPLOAD_PROJ_NAME="pendulum_br_analysis"
LOAD_BR="False"
WHICH_ENV="my_pendulum"
IS_LEAGUE="False"
USE_MIRROR="False"
EVAL_PROT="True"

# Create logs directory if it doesn't exist
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
#conda activate mujoco_sb3_parallel
# Run NUM_WORKERS copies of br_worker in parallel
for i in $(seq 1 ${NUM_WORKERS}); do
    echo "Starting br_worker instance ${i}..."
    nohup python "${BR_WORKER_PATH}" \
	--eval_prot "${EVAL_PROT}" \
        --eval_only "${EVAL_ONLY}" \
        --proj_name "${PROJ_NAME}" \
        --analysis_upload_proj_name "${ANALYSIS_UPLOAD_PROJ_NAME}" \
        --load_br "${LOAD_BR}" \
        --which_env "${WHICH_ENV}" \
        --is_league "${IS_LEAGUE}" \
        --use_mirror "${USE_MIRROR}" \
	--num_brs "${NUM_BRS}" \
	--DEBUG "${DEBUG}" \
        > "${LOGS_DIR}/br_worker_${i}.log" 2>&1 &
    
    echo "br_worker instance ${i} started with PID $!"
done

echo "Started ${NUM_WORKERS} br_worker instances."
echo "Logs are being written to: ${LOGS_DIR}/"
echo "To stop all workers, create a STOP file in the task directory or kill the processes."
