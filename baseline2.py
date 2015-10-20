import csv
import sys, os

from wand.image import Image

DATA_PATH = "../project/better/LISA_TS"
LABEL_FILE = "../project/better/LISA_TS/allAnnotations.csv"
NUM_ITERATIONS = 10

def featureExtractor(imagePath):
	#thresholds
	bt = 10
	gt = 10
	rt = 0

	rawpixels = []
	im  = Image(filename = imagePath)
	w, h = im.width, im.height
	blob = im.make_blob(format='RGB')
	for cursor in range(0, w*h*3,3):
		rawpixels.append((blob[cursor], blob[cursor+1], blob[cursor+2]))

	featureVec = {}
	for i in range(0, len(rawpixels)):
		(r,g,b) = rawpixels[i]
		#if b < bt and g < gt and r > rt:
		featureVec[(r,g,b)] = 1

	return featureVec

def dotProduct(v1, v2):
	common_nonzero_indices = [index for index in v1 if index in v2]
	return sum([v1[index]*v2[index] for index in common_nonzero_indices])

def increment(v1, scale, v2):
	for elem in v2:
		v1[elem] += (scale * v2[elem])

def evaluate(examples, classifier):
	error = 0
	for x, y in examples:
		if classifier(x) != y:
			error += 1

	return float(error)/len(examples)

def SGD(trainExamples, testExamples):
	weights = {}  # feature => weight
	def grad(weights, trainExample):
		x = trainExample[0]
		y = trainExample[1]
		features = featureExtractor(x)
		features_scaled_y = {}
		for feature in features:
			features_scaled_y[feature] = features[feature]*y
		if dotProduct(weights, features_scaled_y) < 1:
			for value in features:
				features[value] *= -y
			return features
		else:
			return {}

	numIters = NUM_ITERATIONS
	for i in range(numIters):
		step_size = 0.00225
		for trainExample in trainExamples:
			gradient = grad(weights, trainExample)
			increment(weights, -step_size, gradient)

		trainError = evaluate(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
		print trainError
		# testError = evaluatePredictor(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
		# print trainError, testError
	return weights

def get_image_labels():
	label_tuples = []
	with open(LABEL_FILE, 'rb') as labels_file:
		labelreader = csv.reader(labels_file, dialect='excel')
		counter = 0
		for row in labelreader:
			print row
			if counter == 0:
				counter += 1
				continue
			if counter > 7000:
				return label_tuples
			line = row[0]
			split_line = line.split(';')
			label = -1
			if split_line[1] == "stop":
				label = 1
			label_tup = (split_line[0], label)
			label_tuples.append(label_tup)
			counter += 1
			
	return label_tuples


def main():
	trainExamples = get_image_labels()
	testExamples = []
	#SGD(trainExamples, testExamples)
	print len(trainExamples)
	for elem in trainExamples:
		print featureExtractor(os.path.join(DATA_PATH,elem[0]))

main()
