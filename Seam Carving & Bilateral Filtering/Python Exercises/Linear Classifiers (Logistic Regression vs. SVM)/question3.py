import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats


#Read the dataset
data = pd.read_csv('crabdata.csv', header = None)

#extracting the classes from the dataset and setting them to values of 1 and 0 instead of 1 and 2
sp = data.iloc[:,0] - 1
sex = data.iloc[:,1] - 1
index = data.iloc[:,2] - 1

#extracting the continous features describing the crabs
data = data.iloc[:,3:]
data = stats.zscore(data)
data = pd.DataFrame(data)

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(data, sex, test_size=0.25, random_state = 1)

#Initializing logistic regression model
logreg = LogisticRegression()
#training the model
logreg.fit(X_train, y_train)
#getting predictions from the trained logistic regression model
y_predictions = logreg.predict(X_test)
#forming the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_predictions)

print('Logistic Regression Confusion Matrix: \n',confusion_matrix)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, y_predictions))
print("Logistic Regression Precision:",metrics.precision_score(y_test, y_predictions))
print("Logistic Regression Recall:",metrics.recall_score(y_test, y_predictions))

#Initializing the SVM model
svm = SVC()
#training the svm model
svm.fit(X_train, y_train)
#getting predictions from the trained svm model
y_predictions = svm.predict(X_test)
#forming the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_predictions)

print('SVM Confusion Matrix: \n',confusion_matrix)
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_predictions))
print("SVM Precision:",metrics.precision_score(y_test, y_predictions))
print("SVM Recall:",metrics.recall_score(y_test, y_predictions))

