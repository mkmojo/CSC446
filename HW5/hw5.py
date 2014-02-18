#This can make sure that division / will not round to the nearest integer.
from __future__ import division
import numpy as np

__author__ = 'qqiu'

# This is global dict used for converting strings to numbers.
# so that the svm can work with the string data.
encode = {'y':1, 'n':-1, '?':0, 'republican':1, 'democrat':-1}

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
def parse_data(filename):
    #parse data, convert strings into numbers.
    lines = open(filename,'r')

    X = []
    y = []
    for i, line in enumerate(lines):
        if i > 16:
            line = line.split()
            line = line[0].split(',')
            y.append( convert_to_number(line[0].split()) )
            X.append( convert_to_number(line[1:]) )

    print "Shuffle Input Data ..."
    X = np.asmatrix(X)
    y = np.asarray(y)
    m = X.shape[0]
    rand_indices = np.random.permutation(m)
    X = X[rand_indices[:]]
    y = y[rand_indices[:]]


    print "Get Training Set ..."
    #348 data points for training
    fea = X[0:348]
    res = y[0:348]

    print "Get Dev Set ..."
    #45 data points for developing
    dev_fea = X[348:348+45]
    dev_res = y[348:348+45]

    print "Get Test Set ..."
    #42 data points for testing
    test_fea = X[348+45:348+45 + 42]
    test_res = y[348+45:348+45 + 42]
    return fea,res, dev_fea, dev_res, test_fea, test_res

# Input     : Number of iterations used to train the svm model
# Output    : W representing the svm model.
# Behavior  : This function trans the svm model. It iterates through the data points in data set X.
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


    #!!! Important !!!
    #Tear W apart for the update rule resulting from gradient descent is different for
    #The coefficients of X and that of bias b
    W_ = W[1:]
    b = W[0].reshape((1,1))
    for t in xrange(n_iter):
        learning_rate = constant/(t+1)
        for i in xrange(N):
            x_i_ = X[i,1:]
            y_i = y[i].item((0, 0)) # make y_i from one element mat to scalar

            if 1 - y_i * (x_i_ * W_ + b) > 0:
                W_ = W_- learning_rate * (1/N * W_ - C * y_i * x_i_.T)
                b = b + learning_rate * C * y_i
            else:
                W_ = W_ - learning_rate * (1/N * W_)
                #b = b
    return np.r_[b, W_]

# Input     : W     --> Model used to make predictions.
#             X     --> Features
#             y     --> Labels
# Output    : accu  --> Accuracy on given dataset X.
# Behavior  : This function takes in a SVM model W and make predictions based on the model. Outputs the final
#             accuracy on the test set.
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
    accu_final = 0
    for exp in xrange(3):
        print "Experiment: ", exp + 1
        print "Parse Data ..."
        fea, res, dev_fea, dev_res, test_fea, test_res = parse_data('voting2.dat')

        print "Training ... "
        n_iters =  [2, 5, 8, 10, 12, 15]
        constants =  [0.2, 0.4, 0.8, 1]
        Cs = [0.03, 0.3, 3, 10, 20, 30, 50, 70, 90, 120, 150, 180]
        accu_max = 0
        for id_iter, n_iter in enumerate( n_iters ):
            for id_constant, constant in enumerate(constants):
                for id_C, C in enumerate(Cs):
                    num_tot_exps = len(n_iters) * len(constants) * len(Cs)
                    num_cnt_exps = (id_iter) * len(constants) * len(Cs) + (id_constant) * len(Cs) + (id_C) + 1
                    if num_cnt_exps % int(num_tot_exps/10) == 0:
                        print "%.2f" % (num_cnt_exps / num_tot_exps * 100), "%"
                    W = train_svm(fea, res, int(n_iter), float(constant), float(C))

                    accu = test_svm(W, dev_fea, dev_res)
                    if accu > accu_max:
                        n_iter_best = n_iter
                        constant_best = constant
                        C_best = C
                        accu_max = accu
                        W_best = W
        print "Test result on dev set with: n_iter = ", n_iter_best, "constant = ", constant_best, "C = ", C_best
        accu0, num_mistake0 = test_svm(W_best, dev_fea, dev_res)
        print "Number of mistakes      :", num_mistake0
        print "Accuracy                :%.2f" % (accu0*100), "%"

        if accu0 > accu_final:
            accu_final = accu0
            W_final = W_best
            exp_final = exp + 1
    #Output the final choice of W and the corresponding accuracy
    print "\nResult on test set with the W from exp :", exp_final
    accu1, num_mistake1 = test_svm(W_final, test_fea, test_res)
    print "Number of mistakes      :", num_mistake1
    print "Accuracy                :%.2f" % (accu1*100), "%"

if __name__ == "__main__":
    main()
