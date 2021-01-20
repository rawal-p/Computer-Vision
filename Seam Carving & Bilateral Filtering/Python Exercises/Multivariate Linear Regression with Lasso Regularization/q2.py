from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def lasso_regression(data, targets, alpha):
    #Fit the model
    lasso = Lasso(alpha=alpha,max_iter=1e5)
    lasso.fit(data,targets)
    w = lasso.coef_
    return w


def RMS_error(predictions,target,num1):

    err = predictions - target
    sse = (np.sum(np.square(err))) / num1
    RMS_err = np.sqrt(sse)

    return RMS_err

#Read the training and testing datasets
train  = pd.read_csv('musicdata.txt', sep = " ",header = None)
test = pd.read_csv('musictestdata.txt', sep = " ", header = None)

#normalize both datasets 
train = stats.zscore(train)
test = stats.zscore(test)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

#Initializing number of training and testing samples 
train_num = train.shape[0]
test_num = test.shape[0]

#Extracting the target values from the training set
train_targets = train.iloc[:,0]

#Extracting the feature values from the training set
train_data = train.iloc[:,1:]

#Extracting the target values from the test set
test_targets = test.iloc[:,0]

#Extracting the feature values from the test set
test_data = test.iloc[:,1:]

#initializing the indexing variable for RMS test and training error
acc = 0

#initializing the lamda values for lasso regression
lamda = np.linspace(1e-5,0.25,1000)

#initializing the lists for storing the error values nonzero weights 
#for each lamda in lasso regression
RMS_train = np.zeros((len(lamda),1))
RMS_test = np.zeros((len(lamda),1))
nonzero_weights = np.zeros((len(lamda),1))

for lam in lamda:

    #using the lasso regression function to create the weights/coefficients vector
    weights = lasso_regression(train_data,train_targets,lam)

    #getting a count of nonzero values in the weights vector
    nonzero_weights[acc] = np.count_nonzero(weights)
    
    #calculating the training predictions from the weights vector 
    train_predictions = train_data.dot(weights)

    #calculating the error between the predicted and actual training targets
    RMS_train[acc] = RMS_error(train_predictions, train_targets,train_num)

    #calculating the test predictions from the weights vector from before
    test_predictions = test_data.dot(weights)
    
    #calculating the error between the predicted and actual test targets
    RMS_test[acc] = RMS_error(test_predictions, test_targets,test_num)    

    acc += 1


plt.grid(True, which="both")
plt.plot(np.log(lamda),RMS_train)
plt.title('RMS Training Error vs. log(lamda)')
plt.ylabel('RMS Training Error')
plt.xlabel('log(lamda)')
plt.show()

minTest = np.amin(RMS_test)
minTest_ind = np.argmin(RMS_test)

plt.grid(True, which="both")
plt.plot(np.log(lamda),RMS_test)
plt.plot(np.log(lamda[minTest_ind]),minTest,'bs')
plt.text(np.log(lamda[minTest_ind]),minTest,'%.3f,%.3f' % (np.log(lamda[minTest_ind]),minTest),horizontalalignment='center',verticalalignment='top')
plt.title('RMS Test Error vs. log(lamda)')
plt.ylabel('RMS Test Error')
plt.xlabel('log(lamda)')
plt.show()

plt.grid(True, which="both")
plt.plot(np.log(lamda),nonzero_weights)
plt.plot(np.log(lamda[minTest_ind]),nonzero_weights[minTest_ind],'bs')
plt.text(np.log(lamda[minTest_ind]),nonzero_weights[minTest_ind],'%.3f,%.3f' % (np.log(lamda[minTest_ind]), nonzero_weights[minTest_ind]),horizontalalignment='left',verticalalignment='bottom')
plt.title('Number of Nonzero Weights vs. log(lamda)')
plt.ylabel('Number of Nonzero Weights')
plt.xlabel('log(lamda)')
plt.show()