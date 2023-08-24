#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M               # memory per node
srun -p gpu.stu --gres gpu:1 ./julia test.jl