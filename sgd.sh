#!/bin/bash

# Shell script to run SGD on image segments.

# Name the job in Grid Engine
#$ -N seg_sgd

# Tell grid engine to use current directory
#$ -cwd

# Tell Grid Engine to notify job owner if job 'b'egins, 'e'nds, 's'uspended is 'a'borted, or 'n'o mail
#$ -m besan

# Tel Grid Engine to join normal output and error output into one file
#$ -j y

python classify.py
