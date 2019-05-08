#!/bin/bash

################
#
# Setting slurm options
#
################

# lines starting with "#SBATCH" define your jobs parameters

# requesting the type of node on which to run job
#SBATCH --partition short

# telling slurm how many instances of this job to spawn (typically 1)
# # SBATCH --ntasks 1

# setting number of CPUs per task (1 for serial jobs)
# #SBATCH --cpus-per-task <number>

# setting memory requirements (in MB)
#SBATCH --mem-per-cpu 10000

# propagating max time for job to run
##SBATCH --time <days-hours:minute:seconds>
##SBATCH --time <hours:minute:seconds>
# #SBATCH --time <minutes>
#SBATCH --time 2:00:00

# Setting the name for the job
# #SBATCH --job-name <example-job>

# setting notifications for job
# accepted values are ALL, BEGIN, END, FAIL, REQUEUE
#SBATCH --mail-type END

# telling slurm where to write output and error
#SBATCH -o /home/bolensadrien/Documents/RL/out-%j.out
#SBATCH -e /home/bolensadrien/Documents/RL/out-%j.out
# #SBATCH --error error-%j.out

################
#
# copying your data to /scratch
#
################

# create local folder on ComputeNode
scratch=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $scratch 

# copy all your NEEDED data to ComputeNode
cp $HOME/Documents/RL/*.py $scratch
cd $scratch

# dont access /home after this line

# if needed load modules here
# module load <module_name>
	
# if needed add export variables here

################
#
# run the program
#
################
which python
python $scratch/q_learning.py

output=$HOME/Documents/RL/output/
mkdir -p $output
cp -r ./.json $output
cp -r ./.npy $output
# cp ./*.png $output
# mv $output/../out-$SLURM_JOB_ID.out $output

cd

# clean up scratch
rm -rf $scratch
unset scratch
unset output

exit 0
