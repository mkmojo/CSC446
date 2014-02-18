#This can make sure that division / will not round to the nearest integer.
from __future__ import division

__author__ = 'qqiu'

# This is global dict used for converting strings to numbers.
# so that the perceptron can work with the string data.
encode = {'y':1, 'n':-1, '?':0, 'republican':1, 'democrat':-1}

data = []       #input data set
res = []        #training set
fea = []        #training set
dev_res = []    #developing set
dev_fea = []    #developing set
test_res = []   #test set
test_fea = []   #test set

import numpy as np
import sys
# Input   : A list of strings
# Output  : A list of numbers
# Behavior: This function takes in feature vector like ['y','n','?'] and convert it into number vector like
#           ['1','-1','0'].
def convert_to_number(inlist):
    outlist = []
    for element in inlist:
        number = encode[element]
        outlist.append(number)
    return outlist

# Input     : void
# Output    : lists res[], fea[], dev_res[], dev_fea[], test_res[], test_fea[].
# Behavior  : This function loads in data from voting2.dat to the lists listed in the above Output section
def parse_data():
    #parse data, convert strings into numbers.
    lines = open('voting2.dat','r')
    for i, line in enumerate(lines):
        line = line.split()
        line = line[0].split(',')
        #348 data points for training
        if i > 16 and i <= 16 + 348:
            res.append( convert_to_number(line[0].split()) )
            fea.append( convert_to_number(line[1:]) )
        #45 data points for developing
        elif i > 16 + 348 and i <= 16 + 348 + 45:
            dev_res.append( convert_to_number(line[0].split()) )
            dev_fea.append( convert_to_number(line[1:]) )
        #42 data points for testing
        elif i > 16 + 348 + 45:
            test_res.append( convert_to_number(line[0].split()) )
            test_fea.append( convert_to_number(line[1:]) )

# Input     : Number of iterations used to train the perceptron model
# Output    : W representing the perceptron model.
# Behavior  : This function trans the perceptron model. It iterates through the data points in data set X.
#             Then updates the W if any mismatch happens.
def train_svm(X, y, n_iter, constant, C):
    X = np.asmatrix(X)
    y = np.asmatrix(y)
    # m is the number of data points
    m = X.shape[0]
    N = m
    # n is the number of features for each data point.
    n = X.shape[1]
    #!!! Important!!!
    #Augment the input data with ones col as the constant offset for linear model.

    #Augment X with one column at the very left; The column is all ones
    X = np.c_[ np.ones(m), np.mat(X)]
    #W is a vector of length n + 1; n is the number of features for each data point
    W = np.zeros(n + 1).reshape(n+1, 1)


    for t in xrange(n_iter):
        learning_rate = constant/(t+1)
        for i in xrange(N):
            x_i = X[i]
            y_i = y[i].item((0, 0)) # make y_i from one element mat to scalar

            if 1 - y_i * x_i * W  > 0:
                W = W - learning_rate * (1/N * W - C * y_i * x_i.T)
            else:
                W = W - learning_rate * (1/N * W)
    return W

def test_svm(W, X, y):
    X = np.asmatrix(X)
    y = np.asarray(y)
    m  = X.shape[0]
    X = np.c_[ np.ones(m), np.mat(X)]
    predictions = np.sign(X * W)
    num_right = np.sum(np.equal(y, predictions))
    num_mistake = m - num_right
    accu = num_right / m
    return accu, num_mistake

def main():
    parse_data()
    print "Parse Data ..."
    print "Training ... "
    #n_iter = int(sys.argv[0])
    #constant = int(sys.argv[1])
    #C = int(sys.argv[2])

    n_iters =  [2, 5, 8, 10, 12, 15]
    constants =  [0.2, 0.4, 0.8, 1]
    Cs = [0.03, 0.3, 3, 10, 20, 30, 50, 70, 90, 120, 150, 180]
    accu_max = 0
    for id_iter, n_iter in enumerate( n_iters ):
        for id_constant, constant in enumerate(constants):
            for id_C, C in enumerate(Cs):
                #print id_iter, id_constant, id_C
                num_tot_exps = len(n_iters) * len(constants) * len(Cs)
                num_cnt_exps = (id_iter) * len(constants) * len(Cs) + (id_constant) * len(Cs) + (id_C) + 1
                print "%.2f" % (num_cnt_exps / num_tot_exps * 100), "%"
                W = train_svm(fea, res, int(n_iter), float(constant), float(C))

                accu = test_svm(W, dev_fea, dev_res)
                #print "Accuracy on dev set is:", accu
                if accu > accu_max:
                    n_iter_best = n_iter
                    constant_best = constant
                    C_best = C
                    accu_max = accu
                    W_best = W
    print "Best result on dev set acquired with: n_iter = ", n_iter_best, "constant = ", constant_best, "C = ", C_best
    accu0, num_mistake0 = test_svm(W_best, dev_fea, dev_res)
    print "Number of mistakes      :", num_mistake0
    print "Accuracy                :%.2f" % accu0, "%"

    print "Result on test set:"
    accu1, num_mistake1 = test_svm(W_best, test_fea, test_res)
    print "Number of mistakes      :", num_mistake1
    print "Accuracy                :%.2f" % accu1, "%"

if __name__ == "__main__":
    main()
