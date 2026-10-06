#!/bin/bash
#SBATCH --account=aip-ajbonner
#SBATCH --partition=gpubase_interac
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --job-name=donut-smoke
#SBATCH --output=logs/smoke_%j.log
#SBATCH --cpus-per-task=8

cd ~/scratch/DONUT-NLL
source ~/scratch/DONUT-NLL/.venv/bin/activate

python train_donut_nll.py \
    --dataset waymo \
    --data_root ~/scratch/DONUT-NLL-og/data \
    --name donut-smoke-test \
    --max_epochs 2 \
    --max_steps 20 \
    --batch_size 2 \
    --devices 1 \
    --num_workers 8 \
    --lambda_smooth 0.05

echo "Smoke test done."
