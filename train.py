import os
import sys
import random
from PIL import Image
import cv2
import util
import numpy as np
import segment_util
"""
Module for training and classifying segments.
"""

"""
Reads the text file that contains the names of the segments that have a stop sign
as per manual labelling.
"""
def read_stop_segments():
	with open('stop_segments.txt', 'r') as f:
		stop_segments_set = set(f.read().splitlines())

	return stop_segments_set

"""
Given the names of files, creates pairs of (the file name, has a stop sign or not).
"""
def label_training_data(files):
	stop_segments_set = read_stop_segments()
	labeled_files = []

	for data_file in files:
		if data_file in stop_segments_set:
			labeled_files.append((data_file, 1))
		else:
			labeled_files.append((data_file, -1))

	return labeled_files

"""
Gets training examples and runs SGD. 
"""
def main(DATA_PATH):

	files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]

	labeled_files = label_training_data(files)

	#separate labeled segments into stop sign and not stop sign lists, shuffle, and downsample
	labeled_files_stop = [x for x in labeled_files if x[1] == 1]
	labeled_files_not = [x for x in labeled_files if x[1] == -1]
	random.shuffle(labeled_files_stop)
	random.shuffle(labeled_files_not)
	labeled_files_not = labeled_files_not[0:500]

	# generate training and test data, append paths to training and test data
	final_with_path = [(DATA_PATH + x[0],x[1]) for x in labeled_files_stop[:len(labeled_files_stop)/2] + labeled_files_not[:len(labeled_files_not)/2]]
	test_with_path = [(DATA_PATH + x[0],x[1]) for x in  labeled_files_stop[len(labeled_files_stop)/2:] + labeled_files_not[len(labeled_files_not)/2:]] 

	weights = util.SGD(final_with_path, test_with_path, segment_util.segmentFeatureExtractor,debug=True,numIters=100)

if __name__ == "__main__":
	main("segmented/RESULTS/")
