#!/bin/bash
#SBATCH -N 1  # Request 1 node
#SBATCH -n 10  # Request 10 tasks (processors)
#SBATCH -o maxsg12_1000it_850samples.out  # Direct output to a custom file

# Source the Conda setup script
source /home/ccollado/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate dmabo

# Install modules (ffmpeg is the one required not installed, which requires gcc too)
module purge
module load intel intel-oneapi-mpi

# Run your Python script
mpiexec -n 10 python dmabo_run.py

# Deactivate conda environment after the script execution completes
conda deactivate
