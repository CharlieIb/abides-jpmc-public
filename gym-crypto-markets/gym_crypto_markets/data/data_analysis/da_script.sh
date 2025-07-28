#!/bin/bash

# --- SBATCH Directives ---
# These directives tell the Slurm scheduler how to allocate resources for your job.
# Adjust these values based on your data size and cluster policies.
#
# Request 1 task (process)
#SBATCH --ntasks=1
# Request 1 CPU core for the task
#SBATCH --cpus-per-task=8
# Request memory (e.g., 8GB - adjust based on your unzipped data size, 1GB zipped could be 10GB+ unzipped)
#SBATCH --mem=64G
# Set maximum job run time (e.g., 1 hour - adjust if your analysis takes longer)
#SBATCH --time=04:00:00
# Name of the job
#SBATCH --job-name=BTC_Qty_Analysis
# Output file for standard output
#SBATCH --output=slurm-%j-analysis.out
# Error file for standard error
#SBATCH --error=slurm-%j-analysis.err


echo "--- Job Started: $(date) ---"
echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Current working directory: $(pwd)"

# --- Module Loading ---
# Purge existing modules to ensure a clean environment
module purge
# Load the base BlueBEAR environment
module load bluebear
# Load the specific bear-apps environment for the Conda module
module load bear-apps/2023a
# Load the Miniforge3 module that provides Conda
module load Miniforge3/24.1.2-0

# --- Conda Environment Setup ---
# Initialize Conda for the current shell session.
# This is crucial for 'conda activate' to work within the script.
source /rds/bear-apps/2023a/EL8-ice/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

# Activate your specific Conda environment.
# Ensure 'ABIDES' is the correct name of your environment.
echo "Activating Conda environment: ABIDES"
conda activate ABIDES

# --- Define Paths ---
# Define the path to your Python analysis script.
ANALYSIS_SCRIPT_PATH="/rds/projects/a/aranboll-ai-research/abides_gym_crypto_sim/abides-jpmc-public/gym-crypto-markets/gym_crypto_markets/data/data_extraction/quantity_analysis.py"

# Define the path to your historical data ZIP file.
HISTORICAL_DATA_ZIP_PATH="/rds/projects/a/aranboll-ai-research/abides_gym_crypto_sim/abides-jpmc-public/gym-crypto-markets/gym_crypto_markets/data/data_extraction/BTCUSDT-trades-2025-05.zip"

# Define the name of the CSV file inside the ZIP.
CSV_FILE_INSIDE_ZIP="BTCUSDT-trades-2025-05.csv"

# --- Update Python Script Configuration (Temporary for this run) ---
# We'll use 'sed' to temporarily update the paths in your Python script
# for this specific Slurm job. This avoids modifying your original Python file.
echo "Updating HISTORICAL_DATA_PATH in Python script for this run..."
sed -i "s|HISTORICAL_DATA_PATH = '.*'|HISTORICAL_DATA_PATH = '${HISTORICAL_DATA_ZIP_PATH}'|" "${ANALYSIS_SCRIPT_PATH}"
sed -i "s|CSV_FILE_IN_ZIP = '.*'|CSV_FILE_IN_ZIP = '${CSV_FILE_INSIDE_ZIP}'|" "${ANALYSIS_SCRIPT_PATH}"

# --- Execute the Python Analysis Script ---
echo "Starting Python analysis script: ${ANALYSIS_SCRIPT_PATH}"
# Use the python executable from the activated Conda environment
python "${ANALYSIS_SCRIPT_PATH}"

# --- Check Exit Status ---
if [ $? -eq 0 ]; then
  echo "Python analysis script completed successfully."
else
  echo "Error: Python analysis script failed. Check slurm-${SLURM_JOB_ID}-analysis.err for details."
fi

# --- Revert Python Script Changes (Optional but Recommended) ---
# It's good practice to revert the changes made by sed to your Python script
# if you only want them for this specific batch run.
# This assumes the original HISTORICAL_DATA_PATH was a placeholder or a different default.
# If you want the path to persist in your Python file, remove these lines.
echo "Reverting changes to Python script..."
sed -i "s|HISTORICAL_DATA_PATH = '${HISTORICAL_DATA_ZIP_PATH}'|HISTORICAL_DATA_PATH = 'path/to/your/historical_trades.zip'|" "${ANALYSIS_SCRIPT_PATH}"
sed -i "s|CSV_FILE_IN_ZIP = '${CSV_FILE_INSIDE_ZIP}'|CSV_FILE_IN_ZIP = 'historical_trades.csv'|" "${ANALYSIS_SCRIPT_PATH}"


echo "--- Job Finished: $(date) ---"
