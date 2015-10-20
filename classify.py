import os
import sys
# import classifier module 

DATA_PATH = ""

def classify_image(path):
	print path
	return 1

def main():
	files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
	for f in files:
		classifier_label = classify_image(os.path.join(DATA_PATH, f))
		print "File: ", f, " Classification: ", classifier_label

if __name__ == "__main__":
	main()

