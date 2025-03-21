#!/bin/bash

# Batch script for running a Python MPI program

#PBS -N MyMPIJob
#PBS -l nodes=3:ppn=4
#PBS -l walltime=01:00:00
#PBS -q batch
#PBS -o outputfile.txt
#PBS -e errorfile.err

# Load the MPI module (uncomment and modify if needed)
# module load mpi

# Activate your Python environment if needed
# source /path/to/your/environment/bin/activate

# Navigate to the directory containing your Python script
cd /path/to/your/python/script

# Run the Python script with mpiexec
mpiexec -n 12 python your_script.py

# Deactivate the Python environment if you activated one
# deactivate
