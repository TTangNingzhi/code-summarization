#!/bin/bash

#$ -q gpu               # Specify queue
#$ -l gpu_card=1        # Specify number of GPU cards to use.
#$ -N tf            # Specify job name

fsync $SGE_STDOUT_PATH &
conda activate torch2.0
cd /afs/crc.nd.edu/user/n/ntang/Project/nztang/code-summarization
python -u transformer.py
