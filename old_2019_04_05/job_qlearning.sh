#!/bin/bash

# #SBATCH --array=0-9
#SBATCH --partition short
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 5000
#SBATCH --time 2:00:00
#SBATCH --job-name q-learning
#SBATCH --mail-type END
#SBATCH -o slurm-%A_%a.out

################
#
# copying your data to /scratch
#
################

# create local folder on ComputeNode
work_dir=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $scratch

# copy all your NEEDED data to ComputeNode
cp $HOME/Documents/RL/*.py $work_dir
cd $work_dir

# dont access /home after this line
module purge
module load intelpython3.2019.0.047
################
#
# run the program
#
################
which python
python q_learning.py $SLURM_ARRAY_TASK_ID

output=$HOME/Documents/RL/output/$1
mkdir -p $output
cp -r ./*.json $output
cp -r ./*.npy $output
# cp ./*.png $output
mv /home/bolensadrien/Documents/RL/out-$SLURM_JOB_ID.out $output

cd

# clean up scratch
rm -rf $scratch
unset scratch
unset output

exit 0
