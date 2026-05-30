#!/bin/bash
#SBATCH --job-name=argo_bgc
#SBATCH --output=results/slurm_%j/slurm.out
#SBATCH --error=results/slurm_%j/slurm.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sphill86@charlotte.edu

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Partition   : $SLURM_JOB_PARTITION"
echo "======================================================"

set -euo pipefail
cd $SLURM_SUBMIT_DIR

RUN_DIR="results/$(date +%Y%m%d_%H%M%S)"
mv results/slurm_${SLURM_JOBID} $RUN_DIR
ln -s $(realpath $RUN_DIR) results/slurm_${SLURM_JOBID}

source ~/envs/ocean/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}

python -u run/main.py --stage all --results_dir $RUN_DIR

echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"

echo ""
echo "======================================================"
echo "  RESULTS SUMMARY"
echo "======================================================"

summarize_losses() {
    local label=$1
    local file=$2
    if [[ -f "$file" ]]; then
        local best
        best=$(tail -n +2 "$file" | awk -F',' 'NR==1 || $3<min {min=$3; row=$0} END {print row}')
        local epoch=$(echo "$best" | cut -d',' -f1)
        local train=$(echo "$best" | cut -d',' -f2)
        local val=$(echo "$best"   | cut -d',' -f3)
        printf "  %-28s  epoch=%-4s  train=%-10s  val=%s\n" "$label" "$epoch" "$train" "$val"
    else
        printf "  %-28s  [not found]\n" "$label"
    fi
}

summarize_losses "Encoder"           "$RUN_DIR/encoder_losses.csv"
summarize_losses "GRU"               "$RUN_DIR/gru_losses.csv"
summarize_losses "Probe (static)"    "$RUN_DIR/probe_static_losses.csv"
summarize_losses "Probe (GRU)"       "$RUN_DIR/gru_probe_losses.csv"
summarize_losses "Probe (raw)"       "$RUN_DIR/probe_raw_losses.csv"
summarize_losses "Probe (baseline)"  "$RUN_DIR/probe_baseline_losses.csv"

echo ""
echo "  Reference (MLP encoder, sequential):"
echo "    Static latent probe       val=0.3273"
echo "    GRU probe                 val=0.3580"
echo "    Raw T/S/O MLP             val=0.3562"
echo "    Depth-only baseline       val=0.8173"
echo "======================================================"
