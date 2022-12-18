#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:30:00
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
#SBATCH -c 2
hostname