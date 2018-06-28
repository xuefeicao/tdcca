#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 3-00:00:00
#SBATCH --mem=16G
#SBATCH --qos=bibs-xl6-condo
#SBATCH --constraint='e5-2670'
module load openblas
module load lapack
stdbuf -oL python data_sim_test_para.py
