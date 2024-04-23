#!/bin/bash

# Name of this job
JOBNAME="rephrasetitleszephyr"

# Number of GPUs you want to use
GPUNUM=1

# The path where this job starts
JOBDIR="/home/kerjan/PDFsToRephrased"

# The path that needs to be added to the environment variable $PATH
# such as the path of Python
MYENV=()
# MYENV+=("/home/kerjan/miniconda3/envs/hurst")  # path 1
# MYENV+=("/home/liac/software/vasp544/bin")    # path 2
# You can add more path in this way

# The folder on the workstation required for the task to run.
# RCP will mount these folders.
MYFOLDER=()
MYFOLDER+=("/home/kerjan/.local")    # folder 1, for python, whole path!!!
MYFOLDER+=("/home/kerjan/miniconda3")    # folder 1, for python, whole path!!!
MYFOLDER+=("/home/kerjan/PDFsToRephrased")    # folder 1, for python, whole path!!!
MYFOLDER+=("/home/kerjan/.cache/huggingface")    # folder 1, for python, whole path!!!
# MYFOLDER+=("/home/liac/software/vasp544/bin")  # folder 2, for additional software
# MYFOLDER+=($JOBDIR)                  # folder 3, for code to run
# You can add more folder in this way

# The commands you want to run in this job
MYCOMMAND=(0)
MYCOMMAND+=("nvidia-smi")       # Command 1
MYCOMMAND+=("source /home/kerjan/miniconda3/bin/activate hurst")        # Command 2
MYCOMMAND+=("python3 transformDataset.py")   # Command 3
# MYCOMMAND+=("sleep 30m")      # Command 4
# You can add more commands in this way