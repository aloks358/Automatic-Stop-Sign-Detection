"""
This class defines an image segmenter.
"""
class ImageSegmenter(object):
	def __init__(self, numSegments, maxIter):
		self.num_segments = numSegments
		self.max_iter = maxIter

	"""
	Actually performs image segmentation on an image.
	"""
	def segment(image):
		def calc_distance(example, center):
        	distance = 0
        	for elem in example:
            	if elem in center:
                	distance += (example[elem] - center[elem])**2
            	else:
                	distance += (example[elem])**2
        	for elem in center:
            	if elem not in example:
                	distance += (center[elem])**2
        	return distance

    	n = len(examples)
    	centroids = [examples[random.randint(0, n - 1)] for k in range(K)]
    	assignments = [None]*n
    	old_cost = None
    	for t in range(self.max_iter):
        	total_cost = 0
        	for i, example in enumerate(examples):
            	min_cost = None
            	assignments[i] = 0
            	for k in range(self.num_segments):
                	cost = calc_distance(example, centroids[k])
                	if min_cost == None or cost < min_cost:
                    	min_cost = cost
                    	assignments[i] = k
            	total_cost += min_cost

        	if total_cost == old_cost:
            	break
        	old_cost = total_cost

        	for k in range(self.num_segments):
            	examples_cluster = [example for i, example in enumerate(examples) if assignments[i] == k]
            	if len(examples_cluster) > 0:
                	first_example = copy.deepcopy(examples_cluster[0])
                	for i in range(1, len(examples_cluster)):
                    	increment(first_example, 1, examples_cluster[i])
                	average = {}
                	for elem in first_example:
                    	average[elem] = float(first_example[elem])/len(examples_cluster)
                    	centroids[k] = average
    	return centroids, assignments, old_cost
