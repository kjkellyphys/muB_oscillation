#!/bin/bash
#SBATCH -J DecayHC
#SBATCH -p shared # specify the shared partition
#SBATCH -n 40     # request 40 cores
#SBATCH -N 1      # request 1 node
#SBATCH -t 1-00:00 # set a time limit for the job, here 12 hours
#SBATCH --mem=20000   # request all memory on the node
#SBATCH -o myjob.out # output file name
#SBATCH -e myjob.err # error file name


python MH_decay_scan_run.py
