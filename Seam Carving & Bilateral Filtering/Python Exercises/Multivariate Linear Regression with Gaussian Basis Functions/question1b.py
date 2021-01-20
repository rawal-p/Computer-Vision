import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats

def gaussianBasis(data,mean,num1,num2):
	features = data.shape[1]
	samples = num2
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


def weightsCalc_l2reg(basis,target,lamda):
	lam_identity = lamda*np.identity(basis.shape[1])
	basis_T = basis.transpose()
	w = np.linalg.inv(lam_identity + basis_T.dot(basis))
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


#Assigning and normalizing the target values
#in the mpg column of the dataset
target_vals = data['mpg']
target_vals_norm = stats.zscore(target_vals)

#getting the number of samples 
samples = len(target_vals)		

#Removing the mpg/target values column from the dataset
data = data.drop(labels='mpg', axis=1)

#Zscore normalization of the dataset
data_norm = stats.zscore(data)

#initializing the lamda coefficients lists
lamda = [0, 0.01, 0.1, 1, 10, 100, 1000]
#lamda = np.linspace(0.01,0.1,7)	

#Number of basis functions to be used for the model
basis_num = 90

#initializing accumulator for indexing the average RMS_test_avg
acc1 = 0

#test error for every Cross Validation fold iteration
RMS_test = np.zeros((10,1))

#average test error for each lamda value
RMS_test_avg = np.zeros((len(lamda),1))

for lam in lamda:

	
	#setting up the 10-fold Cross Validation
	folds = KFold(n_splits=10)

	#initializing accumulator for indexing the Cross Validation iterations
	acc2 = 0

	for train_index, test_index in folds.split(data_norm,np.zeros(shape=(data_norm.shape[0], 1))):
		print('lamda = ',lam) 		
		print('k-iteration: ',acc2)
		data_train, data_test = data_norm[train_index], data_norm[test_index], 
		target_train, target_test = target_vals_norm[train_index], target_vals_norm[test_index]

		#coverting the training and testing data and target lists into 
		#pandas DataFrame to pass to the subsequent functions
		data_train = pd.DataFrame(data_train)
		data_test = pd.DataFrame(data_test)
		target_train = pd.DataFrame(target_train)
		target_test = pd.DataFrame(target_test)

		#randomly sampling training data points for the
		# centers of the gaussian basis functions
		means = data_train.sample(n=basis_num, random_state=1)

		#Creating the gaussian basis functions for the training data
		basis_train = gaussianBasis(data_train,means,basis_num, data_train.shape[0])
		     
		#Creating the weight/coefficients vector
		weights = weightsCalc_l2reg(basis_train,target_train,lam)
		
		#Creating the gaussian basis functions for the test data
		basis_test = gaussianBasis(data_test,means,basis_num,data_test.shape[0])
		
		#Creating the test predictions based on the weights calcualted from the training
		test_predictions = basis_test.dot(weights)
		
		#Calculating the RMS test error using the known target values and the predicted values from the data 
		RMS_test[acc2] = RMS_error(test_predictions, target_test,data_test.shape[0])

		acc2 += 1
	
	#Calculating the average error over the 10 iterations for 
	#each lamda value
	RMS_test_avg[acc1] = np.mean(RMS_test)

	acc1 += 1

#Identifying the minimum RMS test error value and its index
minTest = np.amin(RMS_test_avg)
minTest_ind = np.argmin(RMS_test_avg)
	
print('RMS_test_avg: ', RMS_test_avg)

#Plotting lamda vs. RMS_test_avg
plt.grid(True, which="both")
plt.semilogx(lamda,RMS_test_avg)
#plotting the minimum value calculated from before
plt.semilogx(lamda[minTest_ind],minTest,'bs')	
#assigning a lable to the plotted point above
plt.text(lamda[minTest_ind],minTest,'%.3f,%.3f' % (lamda[minTest_ind],minTest),horizontalalignment='left',verticalalignment='bottom')
plt.title('Average RMS Test Error vs Regularizer value')
plt.ylabel('Average RMS Test Error')
plt.xlabel('Regularizer value (lamda)')
plt.show()