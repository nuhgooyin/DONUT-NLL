#!/bin/bash

#SBATCH --account=aip-ajbonner
#SBATCH --partition=gpubase_interac
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=donut-4gpu-test
#SBATCH --output=logs/4gpu_%j.log
#SBATCH --cpus-per-task=8

cd ~/scratch/DONUT-NLL

source ~/scratch/DONUT-NLL/.venv/bin/activate

python train_donut_nll.py \
    --dataset waymo \
    --data_root ~/scratch/DONUT-NLL-og/data \
    --name donut-4gpu-test \
    --batch_size 2 \
    --max_steps 50 \
    --acc_batch_size 64 \
    --devices 4 \
    --num_workers 8 \
    --lambda_smooth 0.05