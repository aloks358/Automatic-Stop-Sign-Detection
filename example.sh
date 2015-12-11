#!/bin/bash

# Template shell script to run image segmentation on a given image (a path to an image must be added to the last line as a parameter to segment.py).

# Name the job in Grid Engine
#$ -N image_seg

# Tell grid engine to use current directory
#$ -cwd

# Tel Grid Engine to join normal output and error output into one file
#$ -j y

python segment.py
