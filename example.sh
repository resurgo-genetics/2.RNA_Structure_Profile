#!/bin/bash
#SBATCH -n 4                                            # Number of tasks to use, 
#SBATCH -N 1                                            # Number of nodes(machines) to use
#SBATCH -t 0-16:00                                      # Runtime in D-HH:MM, after that this job will be killed by slurm controller
#SBATCH -p normal                                       # Partition to submit to, default is "normal"
#SBATCH --mem=1000                                      # Memory pool for all cores. the unit is MB. this is the upper bond of your memory usage
#SBATCH -o /prog/sbatchExsample/example.out             # File to which STDOUT will be written
#SBATCH -e /prog/sbatchExsample/example.err             # File to which STDERR will be written
#SBATCH --mail-type=FAIL,END                            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=liuzhuo14@mails.tsinghua.edu.cn     # Email to which notifications

source /prog/setup.sh	                                # if the job will use softwares or commands in /prog, you need to add this sentense here.
# use "sbatch example.sh" to submit your job to slurm

<Your shell script here>
