from __future__ import division
import numpy as np
encode = {'y':1, 'n':-1, '?':0, 'republican':1, 'democrat':-1}
lam = 0 #This is the regularization parameter. The larger this is, the
        #more coefficients will be small.
boundry = 0 #This is the decision boundary. Because republican is encoded
            #as 1 and democrat is encoded as -1. This boundary is set to
            #the middle i.e. 0.
"""
Input   : A list of strings
Output  : A list of numbers
Behavior: This function takes in feature vector like ['y','n','?'] and
          convert it into number vector like ['1','-1','0'].
"""
def convert_to_number(inlist):
    outlist = []
    for element in inlist:
        number = encode[element]
        outlist.append(number)
    return outlist

data = []
res = [] # training set
fea = [] # training set
dev_res = [] #developing set
dev_fea = [] #developing set
test_res = [] #test set
test_fea = [] #test set


#parse data, convert strings into numbers for linear regression.
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

########################################################
#Find the parameters for linear model W.

#Turn python type list into np type .mat and .asarray,
#otherwise there will be dimension mismatch ('cannot broadcast' error)
y = np.asarray(res)
rows = len(fea)
cols = len(fea[0])

#!!!Impoartant here!!!
#Add a column of ones to the features, which works as the constant
#coefficient for hypothesis.
X = np.c_[ np.ones(rows), np.mat(fea)]
W = np.linalg.inv(lam*np.eye(cols + 1) + X.T*X) * X.T * y
print W
#Here this W has 16 elements, the w_0 is for all ones that
#works as the constant.

########################################################
#Verify the accuracy on dev data set.
y = np.asarray(dev_res)
rows = len(dev_fea)
X = np.c_[ np.ones(rows), np.mat(dev_fea)]

predicts = []
for i, x_i in enumerate(X):
    predict = x_i * W
    if predict >= boundry:
        p = encode['republican']
    else:
        p = encode['democrat']
    #print predict, p, y[i], p==y[i]
    predicts.append(p)
predicts = np.asarray(predicts).reshape((rows,1))

n_correct = np.sum(np.equal(predicts,y))

accu = n_correct/len(predicts)
print "accuracy for dev set is :", accu , "with lambda:", lam

########################################################
#Verify the accuracy on test data set.
y = np.asarray(test_res)
rows = len(test_fea)
X = np.c_[ np.ones(rows), np.mat(test_fea)]

predicts = []
for i, x_i in enumerate(X):
    predict = x_i * W
    if predict >= boundry:
        p = encode['republican']
    else:
        p = encode['democrat']
#    print predict, p, y[i], p==y[i]
    predicts.append(p)
predicts = np.asarray(predicts).reshape((rows,1))

n_correct = np.sum(np.equal(predicts,y))

accu = n_correct/len(predicts)
print "accuracy for test set is :", accu , "with lambda:", lam
