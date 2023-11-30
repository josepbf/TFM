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
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
#module load cuda/12.2.2
#module load python3/3.6.13
#module load numpy/1.19.5-python-3.6.13-openblas-0.3.13
#module load pandas/1.1.5-python-3.6.13

#module load torch
#module load torchvision
#module load PIL
#module load matplotlib
#module load cython

source ../myVirtEnv/bin/activate

python3 EL_train.py 

deactivate