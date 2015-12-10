import os
import sys
# import classifier module

from PIL import Image
import cv2
import util
import numpy as np
DATA_PATH = "segmented/RESULTS/"

SZ=20
bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
	return img

def hog(img):
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)     # hist is a 64 bit vector
	return hist


def segmentFeatureExtractor(path):
	im = Image.open(path)
	rgb_im = im.convert('RGB')
	featureVec = {}

	s_r = 0
	s_g = 0
	s_b = 0
	c = 0
	for i in range(0, im.size[0]):
		for j in range(0,im.size[1]):
			r, g, b = rgb_im.getpixel((i,j))
			if r > 77 and (r-g) > 17 and (r-b) > 17:
				c += 1
			s_r += r
			s_g += g
			s_b += b
	print s_r
	print s_g
	print s_b
	featureVec["prop_red"] = float(s_r)/(s_r+s_g+s_b)
	featureVec["prop_red_pixels"] = float(c)/(im.size[0]*im.size[1])
	featureVec["num_red_pixels"] = float(c)

	cv_im = cv2.imread(path,0)
	cv_im2 = cv2.resize(cv_im, (100,100))
	d_im2 = deskew(cv_im2)
	hist = hog(d_im2)
	for i in range(0,64):
		featureVec[str(i)] = hist[i]
	return featureVec


def classify_image(path):
	print path
	print segmentFeatureExtractor(path)

def read_stop_segments():
	with open('stop_segments.txt', 'r') as f:
		stop_segments_set = set(f.read().splitlines())

	return stop_segments_set

def label_training_data(files):
	stop_segments_set = read_stop_segments()
	labeled_files = []

	for data_file in files:
		if data_file in stop_segments_set:
			labeled_files.append((data_file, 1))
		else:
			labeled_files.append((data_file, -1))

	return labeled_files

def main():
	## get training data
	## train linear model
	## classify new images

	files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
	files = files[0:100]
	files.append("IMAGES__stop_1323896946.avi_image28.pngtemp3.png")

	labeled_Files = label_training_data(files)

	print files
	print util.SGD(files, None, segmentFeatureExtractor)
	for f in files:
		classifier_label = classify_image(os.path.join(DATA_PATH, f))
		print "File: ", f, " Classification: ", classifier_label

if __name__ == "__main__":
	main()
