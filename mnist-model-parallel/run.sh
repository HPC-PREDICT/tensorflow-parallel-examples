#!/bin/bash -l
#SBATCH --job-name="mnist-data-parallel"
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

module load daint-gpu
module load TensorFlow/1.12.0-CrayGNU-18.08-cuda-9.1-python3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python mnist-model-parallel.py
