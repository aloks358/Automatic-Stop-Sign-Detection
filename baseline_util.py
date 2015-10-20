def getFeatureExtractor():
    def featureExtractor(image):
        featureVec = {}
        for pixel in image:
            if red:
                featureVec[pixel] = 1
        return featureVec
    return featureExtractor

def dotProduct(v1, v2):
    common_nonzero_indices = [index for index in v1 if index in v2]
    return sum([v1[index]*v2[index] for index in common_nonzero_indices])

def increment(v1, scale, v2):
    for elem in v2:
        v1[elem] += (scale * v2[elem])

def SGD(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    numIters refers to a variable you need to declare. It is not passed in.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    def grad(weights, trainExample, featureExtractor):
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

    numIters = 16
    for i in range(numIters):
        step_size = 0.00225
        for trainExample in trainExamples:
            gradient = grad(weights, trainExample, featureExtractor)
            increment(weights, -step_size, gradient)

        # trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        # testError = evaluatePredictor(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        # print trainError, testError
    # END_YOUR_CODE
    return weights
