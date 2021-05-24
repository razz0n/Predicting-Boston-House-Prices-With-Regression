# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:58:34 2021

@author: RaZz oN
"""

from sklearn.datasets import load_breast_cancer

# Loading the dataset 
dataset = load_breast_cancer()

# Load the data into x
x = dataset.data

# Load the target into y
y = dataset.target 


from sklearn.model_selection import train_test_split

# Create train and test data to the dataset
X_train , X_test , y_train , y_test = train_test_split(x, y, random_state=40, test_size=0.2)

# Model Creation and Evaluation

from sklearn.linear_model import LogisticRegression

# Create model for the dataset
lR = LogisticRegression(solver='lbfgs', max_iter=10000)

# Fit the data inot the model
lR.fit(X_train, y_train)

# predict the outcome of the model
y_pred = lR.predict(X_test)

from sklearn.metrics import accuracy_score

# Check the accuracy of the dataset
accuracy_score(y_test, y_pred)

from sklearn.model_selection import cross_val_score

# Applying cross validation for more optimal accuracy measurement
cross_val_score(lR, x, y , cv =10)


"""

Model Evaluation metrics


"""


# =============================================================================
# Confusion matrix
# =============================================================================

from sklearn.metrics import confusion_matrix , classification_report

# Applying confusion matrix for evaluation
conf_matrix = confusion_matrix(y_test,y_pred)

# Provides classification report
class_rep = classification_report(y_test,y_pred)


# =============================================================================
# ROC CURVE
# =============================================================================

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Here, instead of predicting the X_test, we use prediction probability

y_pred = lR.predict_proba(X_test)

# We want only 1 column i.e cancer we reject not cancer column i.e column 0

y_pred = y_pred[:,1]

FPR, TPR, Thresholds = roc_curve(y_test, y_pred)

plt.plot(FPR,TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# =============================================================================
# ROC_AUC SCORE ACCURACY
# =============================================================================
from sklearn.metrics import roc_auc_score

# Check the roc and auc score for the set
roc_auc_score(y_test, y_pred)


# =============================================================================
# Scaling the dataset 
# =============================================================================

from sklearn.preprocessing import MinMaxScaler

sc_data = MinMaxScaler(feature_range=(0,1))

X_train = sc_data.fit_transform(X_train)

X_test = sc_data.fit_transform(X_test)

# 1D array ma convert gareko
y_train = y_train.reshape(-1,1)

y_train = sc_data.fit_transform(y_train)

# =============================================================================
# # Applying the model
# =============================================================================

from sklearn.linear_model import LinearRegression

lR = LinearRegression()

lR.fit(X_train, y_train)


# =============================================================================
# Predicting the values for the model
# =============================================================================

y_pred = lR.predict(X_test)

'''
Since, we have scaled the dataset above into range(0,1). The output predicted 
value will be in between that range. But, the y_test has actual values( not 
scaled) and we need to compare the predicted values (y_pred) with the actual
y_test values. So, we need to transform back the values into the actual format 
using the inverse_transform method.
'''
y_pred = sc_data.inverse_transform(y_pred)


# =============================================================================
# Evaluation Metrics
# =============================================================================

'''

Here “least squares” refers to minimizing the mean squared error
between predictions and expected values.


We are gonna use 5 evaluation metrics such as 

# =============================================================================
# Mean Absolute Error ( MAE ) = ( Σ(1 to n)  ( | y_pred - y_true| ) / n 
# =============================================================================


Mean Absolute Error, or MAE, is a popular metric because, like RMSE, 
the units of the error score match the units of the target value that is
being predicted.

Unlike the RMSE, the changes in MAE are linear and therefore intuitive.


# =============================================================================
# Mean Squared Error ( MSE )= ( Σ(1 to n)  (  y_pred - y_true ) **2 ) / n 
# =============================================================================

This has the effect of “punishing” models more for larger errors when MSE is
used as a loss function. It also has the effect of “punishing” models by
inflating the average error score when used as a metric.

A perfect mean squared error value is 0.0, which means that all
predictions matched the expected values exactly.


# =============================================================================
# Root Mean Squared Error = sqrt ( ( Σ(1 to n)  (  y_pred - y_true ) **2 ) / n 
#         ( RMSE )                               or sqrt(MSE)  
# =============================================================================

Importantly, the square root of the error is calculated, which means that the
units of the RMSE are the same as the original units of the target value
that is being predicted.

For example, if your target variable has the units “dollars,” then the RMSE
error score will also have the unit “dollars” and not “squared dollars” like
the MSE.

Note:  RMSE cannot be calculated as the average of the square root of the mean
squared error values. This is a common error made by beginners and is an
example of Jensen’s inequality.


# =============================================================================
# That is, MSE and RMSE punish larger errors more than smaller errors, 
# inflating or magnifying the mean error score. This is due to the square of 
# the error value. The MAE does not give more or less weight to different types
#  of errors and instead the scores increase linearly with increases in error.
# # =============================================================================



# =============================================================================
# Mean Absolute Percentage Error ( MAPE ) = ( Σ(1 to n)  ( | (y_pred - y_true) /
#                                                         y_true |) / n 
#                                          
# =============================================================================
                                     
      
# =============================================================================
# R2 
# =============================================================================

R-squared (R2) is a statistical measure that represents the proportion of the 
variance for a dependent variable that's explained by an independent variable
or variables in a regression mode

R^2 = 1 - {RSS}/{TSS}
R^2	=	coefficient of determination
RSS	=	sum of squares of residuals
TSS	=	total sum of squares



'''

from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import r2_score

# Applying the mean absolute error evaluation
MAE = mean_absolute_error(y_test, y_pred)

# Applying the mean squared error evaluation
MSE = mean_squared_error(y_test, y_pred)

# Applying the rooted mean squared error evaluation
import math
RMSE = math.sqrt(MSE)

#Applying the r^2 score evaluation
R2 = r2_score(y_test, y_pred)

# Here, we have created a user-defined function to find MAE percentage such 
# that it is easier to read the data performance.
def mean_absolute_percentage_error(y_true, y_pred):
    y_true , y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(abs((y_true - y_pred)/ y_true)) * 100

mean_absolute_percentage_error(y_test, y_pred)


'''

Polynomial Regression Model

'''


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

#Loading the boston dataset
boston = load_boston()

# We want only data of RM which is in 5th column.

#Dividing the dataset into data and target
x = boston.data[:,5]
y = boston.target


# =============================================================================
# 
# No need to normalize and feature-scaling in MLR and PLR 
# 
# =============================================================================


#Splitting the dataset into train and test 
from sklearn.model_selection import train_test_split

X_train , X_test, y_train, y_test = train_test_split(x, y, random_state=40,
                                                     test_size=0.25)


# Applying the polynomial regresssion 

from sklearn.preprocessing import PolynomialFeatures

# =============================================================================
# Generate a new feature matrix consisting of all polynomial combinations of 
# the features with degree less than or equal to the specified degree. 
# For example, if an input sample is two dimensional and of the form [a, b],
# the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
# =============================================================================

poly = PolynomialFeatures(degree = 2)

# Reshaping the X_train into the 1-D array
X_train = X_train.reshape(-1,1)

# Fit and transform the reshaped X_train into the polynomial features
poly_X = poly.fit_transform(X_train)


# Applying Linear Regression

from sklearn.linear_model import LinearRegression

lR = LinearRegression()

# Fitting X-train and y-train in the model  
poly_R = lR.fit(X_train, y_train)

# We need to reshape the X_test as X-train in 1-D array
X_test = X_test.reshape(-1,1)

#Predicting from the X-test 
y_pred = lR.predict(X_test)

# Evaluating the model with r2_score

from sklearn.metrics import r2_score

R2 = r2_score(y_test,y_pred)

'''

Random Forest Regression

'''

from sklearn.ensemble import RandomForestRegressor

# n_estimator defines the no. of decision tree used and max_depth refers to
# the point until which nodes are expanded 
rF = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=88)

# Fitting the model  
rF.fit(X_train,y_train)

#Predicting from X_test
y_pred_rF = rF.predict(X_test)

# Reshaping the scaled data from to 1-D array so that it can be transformed
# back to original data
y_pred_rF = y_pred_rF.reshape(-1,1)

#Applying inverse transform method to transform into original values
y_pred_rF = sc_data.inverse_transform(y_pred_rF)


# =============================================================================
# Evaluating the model 
# =============================================================================

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

MAE = mean_absolute_error(y_test, y_pred_rF)
MSE = mean_squared_error(y_test,y_pred_rF)
R2 = r2_score(y_test, y_pred_rF)

def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(abs((y_true - y_pred)/y_true)) * 100

MAPE = mean_absolute_percentage_error(y_test, y_pred_rF)

'''

Support Vector Regression

'''

from sklearn.svm import SVR

# Kernel defines the algorithm used. Some of 'em are polynomial, gaussian
# rbf, sigmoid and hyperbolic tangent

sVr = SVR(kernel='rbf')

# Fitting the model with X-train and y-train
sVr.fit(X_train,y_train)

#Predicting from X-test
y_pred_sVr = sVr.predict(X_test)

# Reshaping the scaled data from to 1-D array so that it can be transformed
# back to original data
y_pred_sVr = y_pred_sVr.reshape(-1,1)

#Applying inverse transform method to transform into original values
y_pred_sVr = sc_data.inverse_transform(y_pred_sVr)

# =============================================================================
# Evaluating model
# =============================================================================

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Calculating the mean absolute error
MAE = mean_absolute_error(y_test, y_pred_sVr)

#Calculating the mean squared error
MSE = mean_squared_error(y_test,y_pred_sVr)

#Calculating the r^2 score
R2 = r2_score(y_test, y_pred_sVr)

# Creating a user defined function to calculate mean absolute percentage error
def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(abs((y_true - y_pred)/y_true)) * 100

# Calculating the mean absolute percentage error
MAPE = mean_absolute_percentage_error(y_test, y_pred_sVr)