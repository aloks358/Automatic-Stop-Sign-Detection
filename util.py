"""
This module defines a series of general functions.
"""
import random, math
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

def computeR(trainExamples, featureExtractor, weights):
    mean = sum([x[1] for x in trainExamples])/float(len(trainExamples))
    SStot = sum([math.pow(float(x[1] - mean),2) for x in trainExamples])
    pred = [dotProduct(featureExtractor(x[0]),weights) for x in trainExamples]
    for i in range(0,len(pred)):
        if pred[i] > 0:
            pred[i] = 1.0
        else:
            pred[i] = -1.0
    SSres = sum([math.pow(float(trainExamples[i][1] - pred[i]),2) for i in range(0,len(trainExamples))])
    return float(1) - float(SSres)/SStot

def SGD(trainExamples, testExamples, featureExtractor, numIters=10, stepSize=0.00225, debug=False):
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
        #print "Begin iteration " + str(i)
        random.shuffle(trainExamples)
        for trainExample in trainExamples:
            #print str(trainExample)
            gradient = grad(weights, trainExample)
            step = float(1)/math.sqrt(i+1)
            increment(weights, -step, gradient)
        if debug:
            trainError = evaluate(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
            print "weights are: " + str(weights)
            if testExamples == None:
                 print 'Train error: ' + str(trainError)
                 continue
            testError = evaluate(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
            print 'Train error: ' + str(trainError) + ', Test error: ' + str(testError)

    testExamplesPos = [x for x in testExamples if x[1] == 1]
    testExamplesNeg = [x for x in testExamples if x[1] == -1]
    pos  = 'Positive classification error ' + str(evaluate(testExamplesPos, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1 )))
    neg = 'Negative classification error ' + str(evaluate(testExamplesNeg, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1 )))
    beta = 'Weights ' + str(weights)
    r2 = 'R2 ' + str(computeR(trainExamples, featureExtractor, weights))
    f = open('linreg.out', 'w')
    f.write(pos)
    f.write(neg)
    f.write(beta)
    f.write(r2)
    f.close()
    return weights


def logSGD(trainExamples, testExamples, featureExtractor, numIters=10, stepSize=0.00225, debug=False):
    weights = {}  # feature => weight
    eta = 0.1
    def grad(weights, trainExample):
        x = trainExample[0]
        y = trainExample[1]
        features = featureExtractor(x)
        dot = dotProduct(weights,features)
        val = float(-2) * (math.pow(( 1 + math.exp(-1*dot)),-1) - y) * (math.pow((1 + math.exp(-1*dot)),-2))*math.exp(-1*dot)
        for elem in features:
            features[elem] *= val
        return features

    for i in range(numIters):
        random.shuffle(trainExamples)
        for trainExample in trainExamples:
            gradient = grad(weights, trainExample)
            increment(weights, -stepSize, gradient)
        if debug:
            trainError = evaluate(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
            print weights
            if testExamples == None:
                 print 'Train error: ' + str(trainError)
                 continue
            testError = evaluate(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
            print 'Train error: ' + str(trainError) + ', Test error: ' + str(testError)
    return weights
