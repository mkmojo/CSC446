__author__ = 'qqiu'

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
