"""
This module defines a series of general functions.
"""
import random, math

"""
Helper functions for computing stochastic gradient descent
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

"""
Wrapper function that allows for cache-ing of feature extractions
"""
cache = {}
def fe(featureExtractor,x):
    if x in cache:
        return cache[x]
    else:
        res = featureExtractor(x)
        cache[x] = res 
        return res

"""
Helper method that computes the R2 statistic, or the coefficient of
determination, which gives an estimate of the amount of variance
explained by a given linear model. R2 = 1 - SSres/SStot, where 
SStot = sum(yi - ybar)^2 and SSres = sum(predicted_i - yi)^2
"""
def computeR(trainExamples, featureExtractor, weights):
    mean = sum([x[1] for x in trainExamples])/float(len(trainExamples))
    print mean
    SStot = sum([math.pow(float(x[1] - mean),2) for x in trainExamples])
    pred = [dotProduct(fe(featureExtractor,x[0]),weights) for x in trainExamples]
    for i in range(0,len(pred)):
        if pred[i] > 0:
            pred[i] = 1.0
        else:
            pred[i] = -1.0
    SSres = sum([math.pow(float(trainExamples[i][1] - pred[i]),2) for i in range(0,len(trainExamples))])
    return float(1) - float(SSres)/SStot


def writeOutFile(trainExamples, testExamples, featureExtractor, weights):
    trainError = evaluate(trainExamples, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1))
    testError = evaluate(testExamples, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1))
    trainExPos = [x for x in trainExamples if x[1] == 1]
    trainExNeg = [x for x in trainExamples if x[1] == -1]
    testExamplesPos = [x for x in testExamples if x[1] == 1]
    testExamplesNeg = [x for x in testExamples if x[1] == -1]
    def classify(x,weights):
        if dotProduct(fe(featureExtractor,x), weights) >= 0: return 1
        return -1
    residuals = [(testExamples[i][1] -classify(testExamples[i][0],weights))^2 for i in range(len(testExamples))]
    y = [testExamples[i][1] for i in range(len(testExamples))]  
    t_pos_n = 'Positive n is ' + str(len(trainExPos))
    t_neg_n = 'Negative n is ' + str(len(trainExNeg))
    t_pos  = 'Positive classification error ' + str(evaluate(trainExPos, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1 )))
    t_neg = 'Negative classification error ' + str(evaluate(trainExNeg, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1 )))
    pos_n = 'Positive n is ' + str(len(testExamplesPos))
    neg_n = 'Negative n is ' + str(len(testExamplesNeg))
    pos  = 'Positive classification error ' + str(evaluate(testExamplesPos, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1 )))
    neg = 'Negative classification error ' + str(evaluate(testExamplesNeg, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1 )))
    beta = 'Weights ' + str(weights)
    r2 = 'R2 ' + str(computeR(trainExamples, featureExtractor, weights))
    f = open('sgd_linreg.out', 'w')
    f.write('train' + '\n')
    f.write(t_pos_n + '\n')
    f.write(t_neg_n + '\n')
    f.write(t_pos + '\n')
    f.write(t_neg + '\n')
    f.write('test' + '\n')
    f.write(pos_n + '\n')
    f.write(neg_n + '\n')
    f.write(pos + '\n')
    f.write(neg + '\n')
    f.write(beta + '\n')
    f.write(str(trainError) + '\n')
    f.write(str(testError) + '\n')
    f.write(r2 + '\n')
    f.write(str(residuals) + '\n')
    f.write(str(y) + '\n')
    f.close()
    g = open('weights.out','w')
    g.write(str(weights))
    g.close()

"""
Function for computing a linear classifier using the method of stochasitc gradient descent. Generates model,
computes errors and relevant statistics, and writes them to an out file"
"""
def SGD(trainExamples, testExamples, featureExtractor, numIters=10, stepSize=0.00225, debug=False):
    weights = {}  # feature => weight
    features = {}
    def grad(weights, trainExample):
        x = trainExample[0]
        y = trainExample[1]
        features = fe(featureExtractor,x)
        if y*dotProduct(weights, features) < 1:
            for value in features:
                features[value] *= -y
            return features
        else:
            return {}
    
    for i in range(numIters):
        random.shuffle(trainExamples)
        for trainExample in trainExamples:
            gradient = grad(weights, trainExample)
            step = float(1)/math.sqrt(i+1)
            increment(weights, -step, gradient)
        if debug:
            trainError = evaluate(trainExamples, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1))
            testError = evaluate(testExamples, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1))
            print 'Train error: ' + str(trainError) + ', Test error: ' + str(testError)

    writeOutFile(trainExamples,testExamples,weights)
    return weights

""" 
Function for computing a linear classifier with L2 norm penalty using the method of stochasitc gradient descent. 
Generates model, computes errors and relevant statistics, and writes them to an out file"
"""

##we let the lambda value be some constant
lam = 0.07096536
def regSGD(trainExamples, testExamples, featureExtractor, numIters=10, stepSize=0.00225, debug=False):
    def grad(weights, trainExample, i, u,v):
        w = {}
        x = trainExample[0]
        y = trainExample[1]
        features = fe(featureExtractor,x)
        ymin_wtphit = y - dotProduct(weights, features)
        step = float(1)/math.sqrt(i+1)
        increment(u,float(stepSize)*(lam - ymin_wtphit)*(-1.0),features)
        increment(v,float(stepSize)*(lam + ymin_wtphit)*(-1.0),features)
        for elem in u:
            if u[elem] < 0: u[elem] = 0
            if v[elem] < 0: v[elem] = 0
            w[elem] = u[elem] - v[elem]
        return w

    weights = {}  # feature => weight
    u = {}
    v = {}
    features = {}
    for i in range(numIters):
        random.shuffle(trainExamples)
        for trainExample in trainExamples:
            weights = grad(weights, trainExample, i,u,v)
        if debug:
            trainError = evaluate(trainExamples, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1))
            testError = evaluate(testExamples, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1))
            print 'Train error: ' + str(trainError) + ', Test error: ' + str(testError)
    writeOutFile(trainExamples,testExamples,weights)
    return weights

