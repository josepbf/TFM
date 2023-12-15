#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J firstjob
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 3:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
#module load gcc/10.3.0-binutils-2.36.1
#module load gcc/12.3.0-binutils-2.40
#module load sqlite3/3.42.0
#module load python3/3.11.4
#module load cython/0.29.35-python-3.11.4
#module load cuda/11.6

source ../myVirtEnv/bin/activate

/appl/python/3.11.4/bin/python3.11 EL_train.py 

deactivate