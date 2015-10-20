from __future__ import print_function
import cv2
import sys
# import classifier module

def run_classifier(frame):
	print("Run classifier on frame")
	classification = 0
	return classification

def main():
	found_stop_sign = False
	if len(sys.argv) > 1:
		video_path = sys.argv[1]
		video_capture = cv2.VideoCapture(video_path)

		while True:
			ret, frame = video_capture.read()
			label = run_classifier(frame)
			if label == 1:  # stop sign in video
				found_stop_sign = True
				break

		if found_stop_sign:
			print("Found stop sign")
		else:
			print("Didn't find stop sign")

	else:
		print("Must provide the path to the video as an argument")

if __name__ == "__main__":
	main()

