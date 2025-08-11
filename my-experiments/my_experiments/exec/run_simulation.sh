#!/bin/bash

#SBATCH --output=slurm-%j.out  # Ensure you capture stdout
#SBATCH --error=slurm-%j.err    # Ensure you capture stderr
#SBATCH --ntasks=4
#SBATCH --time=7-00:00:00
# Add other SBATCH directives you use (e.g., --nodes, --ntasks-per-node, --time, --partition)

module purge; module load bluebear
module load bear-apps/2023a
module load Miniforge3/24.1.2-0

# Initialize Conda (This path is for the system's Miniforge, which provides the 'conda' command)
source /rds/bear-apps/2023a/EL8-ice/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

# --- Define the CORRECT path to your ABIDES environment's Python executable ---
YOUR_ACTUAL_ABIDES_ENV_PYTHON_PATH="/rds/homes/c/cai481/.conda/envs/ABIDES/bin/python"


echo "--- PRE-RUN DIAGNOSTICS ---"
echo "Attempting to run direct Python diagnostic from ABIDES environment:"

# Use the actual path to your ABIDES environment's Python for the diagnostic
"${YOUR_ACTUAL_ABIDES_ENV_PYTHON_PATH}" -u -c "$(cat << 'EOF_PYTHON_DIAGNOSTIC'
import sys;
print('sys.executable:', sys.executable);
print('sys.path:', sys.path);
try:
    import yaml;
    print("YAML imported successfully within ABIDES env's python");
except ImportError:
    print("YAML import FAILED within ABIDES env's python");
EOF_PYTHON_DIAGNOSTIC
)"

echo "--- END PRE-RUN DIAGNOSTICS ---"


cd /rds/projects/a/arnaboll-ai-research/projects/abides_gym_crypto_sim/abides-jpmc-public/my-experiments/my_experiments/exec

echo "Starting the gym simulation..."

# --- CRUCIAL CHANGE HERE: Use the actual path to your ABIDES environment's Python ---
# This is the line that runs your main simulation script.
"${YOUR_ACTUAL_ABIDES_ENV_PYTHON_PATH}" -u run_gym_simulation.py base_config_bear.yaml --mode train-abides

if [ $? -eq 0 ]; then
  echo "Gym simulation completed successfully."
else
  echo "Error: Gym simulation failed. Check the output above for details."
fi