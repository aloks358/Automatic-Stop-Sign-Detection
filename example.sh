#!/bin/bash

# Shell script to run image segmentation on a given image.

# Name the job in Grid Engine
#$ -N image_seg

# Tell grid engine to use current directory
#$ -cwd

# Tel Grid Engine to join normal output and error output into one file
#$ -j y

python segment.py
