#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 3-00:00:00
#SBATCH --mem=24G
#SBATCH --qos=bibs-xl6-condo
module load R/3.4.2
stdbuf -oL Rscript compare_scca.R
