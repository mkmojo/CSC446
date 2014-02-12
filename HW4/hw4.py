#This can make sure that division / will not round to the nearest integer.
from __future__ import division
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

# Input   : A number
# Output  : 1 or -1
# Behavior: This function takes in a value and decided which class this value belongs to. This is the f used in equation
#           (4.52) in Bishop.
def f(a):
    if a >= 0:
        return 1
    else:
        return -1


# Input   : W     --> the weights for the perceptron.
#           x_n   --> one data point.
# Output  : Group the data point x_n belongs to with current weights W.
# Behavior: This function takes in one data point in given data set, uses currrent perceptron model do predict its group.
#           It give back the model W's perspective of what group x_n belongs to.
def get_prediction(W, x_n):
    return f(x_n * W.T)


# Input   : W     --> the weights of the perceptron.
#           x_n   --> one data point.
#           t_n   --> actual group x_n belongs to.
# Output  : a number.
# Behavior: This function is a helper function for get_err(W, X, y). It calculates the penalty contribution for
#           mis-calssifying the data point x_n.
def get_err_fraction(W, x_n, t_n):
    return x_n * W.T * t_n

# Input   : x     --> a number
# Output  : the absolute value of x.
# Behavior: Return the absolute value of x.
def abs(x):
    if x < 0:
        return -x
    else:
        return x


# Input   : W     --> the weights of the perceptron.
# Output  : X     --> the voting data sets (with features converted to numbers).
# Behavior: Calculate all the points that are mis-classified. This will go through the entire matrix to get the err
#           because adjusted W could cause right answer to be wrong under new current W.
#           However this will converge if there is the data is linearly separable.
def get_err(W, X, y):
    N = X.shape[0]
    err = 0.0
    for n in xrange(N):
        x_n = X[n]
        t_n = y[n]
        predict = get_prediction(W, x_n)

        if predict != t_n:
            err_frac = get_err_fraction(W, x_n, t_n)
            err = err - err_frac
    return err

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
def train_perceptron(n_iter):
    y = np.asarray(res)
    n_row = len(fea)
    n_col = len(fea[0])
    #!!! Important!!!
    #Augment the input data with ones col as the constant offset for linear model.
    X = np.c_[ np.ones(n_row), np.mat(fea)]
    W = np.c_[ np.ones(1) , np.zeros(n_col).reshape((1,n_col))]

    W_best = W
    E_min = get_err(W, X, y)
    i = 0
    while i <= n_iter:
        if i %(n_iter / 20) == 0:
           print i / n_iter * 100, "%"
        x_n = X[i%n_row]
        t_n = y[i%n_row]

        if get_prediction(W, x_n) != t_n:
            W = (W.T + x_n.T * t_n).T

        E_cnt = get_err(W, X, y)
        if E_cnt < E_min:
            E_min = E_cnt
            W_best = W
            #print W, E_min

        i = (i + 1)
    return W_best,W


# Input     : W     --> Model used to make predictions.
# Output    : accu  --> Accuracy on test dataset.
# Behavior  : This function takes in a perceptron model W and make predictions based on the model. Outputs the final
#             accuracy on the test set.
def test(W):
    #Verify the accuracy on dev data set.
    y = np.asarray(test_res)
    rows = len(test_fea)
    X = np.c_[ np.ones(rows), np.mat(test_fea)]

    predicts = []
    for i, x_n in enumerate(X):
        predict = get_prediction(W,x_n)
        predicts.append(predict)
    predicts = np.asarray(predicts).reshape((rows,1))

    n_correct = np.sum(np.equal(predicts,y))

    return n_correct/len(predicts), get_err(W, X, y)


__author__ = 'qqy'

data = []
res = []        #training set
fea = []        #training set
dev_res = []    #developing set
dev_fea = []    #developing set
test_res = []   #test set
test_fea = []   #test set

# This is global dict used for converting strings to numbers.
# so that the perceptron can work with the string data.
encode = {'y':1, 'n':-1, '?':0, 'republican':1, 'democrat':-1}

def main():
    parse_data()
    print "Parse Data ..."
    print "Training ... "

    W_best, W_last = train_perceptron(int(sys.argv[1]))
    print "The best model is:", W_best
    print "The last model is:", W_last

    accu_W_best, Err_W_best = test(W_best)
    accu_W_last, Err_W_last = test(W_last)
    print "Accuracy on test set is, best model :", accu_W_best, "with err:", Err_W_best
    print "Accuracy on test set is, last model :", accu_W_last, "with err:", Err_W_last


if __name__ == "__main__":
    main()
