import os
import sys

DATA_PATH = ""
LABEL_FILE = ""

def classify_image(path):
	print path
	return 1

def get_image_labels():

	label_map = {}
	with open(os.path.join(DATA_PATH, LABEL_FILE), 'r') as label_map_file:
		for line in label_map_file:
			split_line = line.split()
			label_map[split_line[0]] = int(split_line[1])

	return label_map

def main():

	files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
	label_map = get_image_labels()

	for f in files:
		correct_label = label_map[f]
		# some sort of feature extraction on (os.path.join(DATA_PATH, f))
		
if __name__ == "__main__":
	main()

