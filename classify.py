import os
import sys
import random
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

def red_squares_partition(im, rgb_im):

	num_col = 10
	max_red = 0
	for i in range(0, num_col):
		for j in range(0, num_col):
			left_x = i * im.size[0]/num_col
			left_y = j * im.size[1]/num_col
			num_red = 0
			for k in range(0, im.size[0]/num_col):
				for l in range(0, im.size[1]/num_col):
					r, g, b = rgb_im.getpixel((left_x + k, left_y + l))
					if r > 77 and (r-g) > 17 and (r-b) > 17:
						num_red += 1
			if num_red > max_red:
				max_red = num_red

	return max_red/(im.size[0]/num_col * im.size[1]/num_col)  # Proportion of max red pixels in a square

def segmentFeatureExtractor(path):
	im = Image.open(path)
	rgb_im = im.convert('RGB')
	featureVec = {}

	s_r = 0
	s_g = 0
	s_b = 0
	c = 0
	bl = 0
	i_vec = []
	r_vec = []

	featureVec["max_red_square_prop"] = red_squares_partition(im, rgb_im)

	for i in range(0, im.size[0]):
		for j in range(0,im.size[1]):
			r, g, b = rgb_im.getpixel((i,j))
			if r == 0 and g == 0 and b == 0:
				bl += 1
				continue
			if r > 77 and (r-g) > 17 and (r-b) > 17:
				c += 1
			s_r += r
			s_g += g
			s_b += b
			r_vec.append(r)
			intensity = r*0.2989 + g*0.5870 + b*0.1140
			i_vec.append(intensity)
	featureVec["r_std"] = np.std(np.array(r_vec), axis = 0)
	featureVec["i_std"] = np.std(np.array(i_vec), axis = 0)
	featureVec["prop_red"] = float(s_r)/(s_r+s_g+s_b)
	featureVec["prop_red_pixels"] = float(c)/(im.size[0]*im.size[1] - bl)
	featureVec["num_red_pixels"] = float(c)

	cv_im = cv2.imread(path,0)
	cv_im2 = cv2.resize(cv_im, (100,100))
	d_im2 = deskew(cv_im2)
	hist = hog(cv_im2)
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

	labeled_files = label_training_data(files)
	labeled_files_stop = [x for x in labeled_files if x[1] == 1]
	labeled_files_not = [x for x in labeled_files if x[1] == -1]
	random.shuffle(labeled_files_stop)
	random.shuffle(labeled_files_not)
	final = labeled_files_stop+ labeled_files_not[0:100]
	random.shuffle(final)
	final_with_path = [(DATA_PATH + x[0],x[1]) for x in final]
	#test_with_path = [(DATA_PATH + x[0],x[1]) for x in labeled_files_stop[50:100]]
	print util.SGD(final_with_path, None, segmentFeatureExtractor,debug=True)
	for f in final_with_path:
		classifier_label = classify_image(f[0])
		print "File: ", f, " Classification: ", classifier_label

if __name__ == "__main__":
	main()
