"""
This class defines an image segmenter.
"""
class ImageSegmenter(object):
	def __init__(self, numSegments, maxIter):
		self.num_segments = numSegments
		self.max_iter = maxIter
		self.features = ["Intensity", "x", "y"]
		self.feature_weights = {"Intensity" : 5, "x": 6, "y": 6}
		self.x = 145
		self.y = 145

	"""
	Allows adjusting of weights to features when comparing distances.
	"""
	def set_weights(new_weights):
		if new_weights.keys() != self.feature_weights.keys():
			raise ValueError("The weights must correspond to the features.")
		self.feature_weights = new_weights

	"""
	Take the current image format (a one-dimensional list of pixels), and extract
	feature information for the pixels into a list
	"""
	def convert_image_to_pixels(image):
		pixels = []
		for i in range(len(image)/self.y):
			for j in range(self.x):
				pixels.append({"Intensity" : intensity_calc(image[i*self.y + j]), "x": j, "y": i})
		return pixels

	"""
	Takes a pixel (a RGB tuple) and returns its intensity (grayscale value)
	"""
	def intensity_calc(pixel):
		intensity = pixel[0]*0.2989 + pixel[1]*0.5870 +pixel[2]*0.1140
		return intensity

	"""
	Actually performs image segmentation on a list of pixels.
	"""
	def segment(image):
		pixels = convert_image_to_pixels(image)
		def calc_distance(pixel, center):
			return sum([((pixel[elem]-center[elem])**2)*self.feature_weights[elem] for elem in pixel])
		n = len(pixels)
		centroids = [pixels[random.randint(0, n - 1)] for k in range(K)]
		assignments = [None]*n
		old_cost = None
		for t in range(self.max_iter):
			total_cost = 0
			for i, pixel in enumerate(pixels):
				min_cost = None
				assignments[i] = 0
				for k in range(self.num_segments):
					cost = calc_distance(pixel, centroids[k])
					if min_cost == None or cost < min_cost:
						min_cost = cost
						assignments[i] = k
				total_cost += min_cost

			if total_cost == old_cost:
				break
			old_cost = total_cost

			for k in range(self.num_segments):
				pixels_cluster = [pixel for i, pixel in enumerate(pixels) if assignments[i] == k]
				if len(pixels_cluster) > 0:
					first_pixel = copy.deepcopy(pixels_cluster[0])
					for i in range(1, len(pixels_cluster)):
						increment(first_pixel, 1, pixels_cluster[i])
					average = {}
					for elem in first_pixel:
						average[elem] = float(first_pixel[elem])/len(pixels_cluster)
						centroids[k] = average
		return centroids, assignments, old_cost
