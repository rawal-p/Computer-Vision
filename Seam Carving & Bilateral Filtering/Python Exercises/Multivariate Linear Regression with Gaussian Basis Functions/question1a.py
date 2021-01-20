import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def gaussianBasis(data,mean,num1,num2):
	#intializing variables
	features = data.shape[1]
	samples = num2
	#initializing final basis matrix (+1 for the bias terms)
	#and covariance matrix
	basis = np.ones((samples, num1+1))
	covariance = 2*np.identity(features)
	for j in range(0, num1+1):
		for i in range(0,samples):
			if j == 0:
				basis[i,j] = 1
			else:
				X = data.iloc[i].to_numpy()
				MU = mean.iloc[j-1].to_numpy()
				A = X - MU
				covariance_inv = np.linalg.inv(covariance)
				A_T = A.transpose()
				temp = A.dot(covariance_inv)
				temp = temp.dot(A_T)
				temp = (-0.5) * temp
				temp = np.exp(temp)
				basis[i,j] = temp
	return basis


def weightsCalc(basis,target):
	basis_T = basis.transpose()
	w = np.linalg.inv(basis_T.dot(basis))
	w = w.dot(basis_T)
	w = w.dot(target)
	return w

def RMS_error(predictions,target,num1):
	err = predictions - target
	sse = (np.sum(np.square(err))) / num1
	RMS_err = np.sqrt(sse)
	return RMS_err



#Read the dataset
data = pd.read_csv('auto-mpg.csv')

#Remove the car name column
data = data.drop(labels='car name', axis=1)


#Assigning and normalizing the target values in the mpg column of the dataset
target_vals = data['mpg']
target_vals_norm = stats.zscore(target_vals)

target_vals_norm = pd.DataFrame(target_vals_norm)

#getting the number of samples, training and testing values
samples = len(target_vals)		
train_num = 100
test_num = samples - train_num


#Assigning the training target values
target_train = target_vals_norm.head(train_num)	

#Assigning the test target values
target_test = target_vals_norm.tail(test_num)

#Remove the car name column
data = data.drop(labels='mpg', axis=1)

#Zscore normalization of the dataset (mean = 0, std = 1) for all the features
data_norm = stats.zscore(data)
data_norm = pd.DataFrame(data_norm)

#Assigning the training data values
data_train = data_norm.head(train_num)

#Assigning the testing data values
data_test = data_norm.tail(samples-train_num)

#initializing the basis_num list
basis_num = np.arange(5,100,10)

#initializing accumulator for indexing the RMS_test and training lists
acc = 0

#initializing lists for storing the RMS training and testing errors 
#for the number of basis functions used
RMS_train = np.zeros((10,1))
RMS_test = np.zeros((10,1))


for mu in basis_num:

	print('mu = ',mu)
	
	#randomly sampling training data points for the
	# centers of the gaussian basis functions
	means = data_train.sample(n=mu, random_state=1)

	#Creating the gaussian basis functions for the training data
	basis_train = gaussianBasis(data_train,means,mu,train_num)
	
	#Creating the weight/coefficients vector
	weights = weightsCalc(basis_train,target_train)

	#Creating the test predictions based on the weights calcualted from the training
	train_predictions = basis_train.dot(weights)
	
	#Calculating the RMS training error using the known target values and the predicted values from the data 
	RMS_train[acc] = RMS_error(train_predictions, target_train,train_num)
	
	#Creating the gaussian basis functions for the test data
	basis_test = gaussianBasis(data_test,means,mu,test_num)
	
	#Creating the test predictions based on the weights calcualted from the training
	test_predictions = basis_test.dot(weights)
	
	#Calculating the RMS test error using the known target values and the predicted values from the data 
	RMS_test[acc] = RMS_error(test_predictions, target_test,test_num)


	acc += 1


print('RMS_test: ', RMS_test)
plt.grid(True, which="both")

plt.plot(basis_num,RMS_train,label="Training")
plt.plot(basis_num,RMS_test,label="Test")
plt.legend(loc="best")
plt.title('RMS Error vs Number of Basis Functions')
plt.ylabel('RMS Error')
plt.xlabel('Number of Bais Functions')
plt.show()
