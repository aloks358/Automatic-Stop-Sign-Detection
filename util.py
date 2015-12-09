"""
This module defines a series of general functions.
"""

def dotProduct(v1, v2):
	common_nonzero_indices = [index for index in v1 if index in v2]
	return sum([v1[index]*v2[index] for index in common_nonzero_indices])

def increment(v1, scale, v2):
	for elem in v2:
		if elem in v1.keys():
			v1[elem] += (scale * v2[elem])
		else:
			v1[elem] = (scale * v2[elem])

def evaluate(examples, classifier):
    error = 0
    for x, y in examples:
        if classifier(x) != y:
            error += 1
    return float(error)/len(examples)

 def SGD(trainExamples, testExamples, numIters=10, stepSize=0.00225, debug=False):
    weights = {}  # feature => weight
    def grad(weights, trainExample):
        x = trainExample[0]
        y = trainExample[1]
        features = featureExtractor(x)
        if y*dotProduct(weights, features) < 1:
            for value in features:
                features[value] *= -y
            return features
        else:
            return {}
    
    for i in range(numIters):
        for trainExample in temp:
            gradient = grad(weights, trainExample)
            increment(weights, -stepSize, gradient)
    
    if debug:
        trainError = evaluate(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        testError = evaluate(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print 'Train error: ' + trainError + ', Test error: ' + testError
    return weights


