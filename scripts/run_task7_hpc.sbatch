#!/bin/bash
#SBATCH --job-name=task7_perm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/perm_importance.slurm.%j.out
#SBATCH --error=logs/perm_importance.slurm.%j.err

# NYU Greene template — adjust the GPU type / partition / module loads
# to match your HPC environment if needed.
# Examples:
#   #SBATCH --gres=gpu:rtx8000:1
#   #SBATCH --gres=gpu:a100:1
#   #SBATCH --partition=gpu

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "[$(date '+%F %T')] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

# --- environment ---
# If you use a Singularity / Apptainer overlay on Greene, replace the next
# block with your overlay invocation. If you have a plain venv/conda, keep
# this template and edit the activate line.
if [ -f "${HOME}/envs/ml1003/bin/activate" ]; then
    source "${HOME}/envs/ml1003/bin/activate"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate ml1003
fi

# Make sure TabPFN's auth token is available if v2.5+ weights need it.
# (cached via tabpfn.browser_auth.save_token or TABPFN_TOKEN env var)
: "${TABPFN_TOKEN:=}"

python -u scripts/task7_permutation_importance.py 2>&1 | tee logs/perm_importance.log
echo "[$(date '+%F %T')] sbatch wrapper done"
