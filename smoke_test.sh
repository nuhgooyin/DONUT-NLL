#!/bin/bash

#SBATCH --account=aip-ajbonner
#SBATCH --partition=gpubase_interac
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=donut-full
#SBATCH --output=logs/full_%j.log
#SBATCH --cpus-per-task=8

cd ~/scratch/DONUT-NLL

source ~/scratch/DONUT-NLL/.venv/bin/activate

python train_donut_nll.py \
    --dataset waymo \
    --data_root ~/scratch/DONUT-NLL-og/data \
    --name donut-full-batch2 \
    --max_epochs 2 \
    --batch_size 2 \
    --max_steps 100 \
    --acc_batch_size 64 \
    --devices 1 \
    --num_workers 8 \
    --lambda_smooth 0.05