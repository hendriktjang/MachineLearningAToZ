#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:34:07 2019

@author: hendrik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test result
y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)

#create a matrix containing the optimum team of variables. So we start with all features of X
# here it means i take all rows of X, with columns 0-5, but why not just X_opt = X[:,:]?
X_opt = X[:, [0,1,2,3,4,5]]

#select a Significant Level (SL) to stay in the model, we choose 5%
#We then need to fit model into X_opt
# we will use a new library to create the model. So we will create a new regressor object
#This new class is OLS (Ordinary Least Square)
#If you control I the documentation, it will say that for X variable, you need 
#to include the intercept, which we already did for 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

#We need to get P values of each predictor and compare it with the SL
regressor_OLS.summary()

""" Result: that means we ned to remove x3
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        2.73e+04   3185.530      8.571      0.000    2.09e+04    3.37e+04
x1           2.73e+04   3185.530      8.571      0.000    2.09e+04    3.37e+04
x2          1091.1075   3377.087      0.323      0.748   -5710.695    7892.910
x3           -39.3434   3309.047     -0.012      0.991   -6704.106    6625.420
x4             0.8609      0.031     27.665      0.000       0.798       0.924
x5            -0.0527      0.050     -1.045      0.301      -0.154       0.049
"""

#Remove X[2]
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

""" Result: that means need to remove X2
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       2.753e+04   3072.973      8.960      0.000    2.13e+04    3.37e+04
x1          2.753e+04   3072.973      8.960      0.000    2.13e+04    3.37e+04
x2          -573.7029   2838.043     -0.202      0.841   -6286.386    5138.981
x3             0.8624      0.030     28.282      0.000       0.801       0.924
x4            -0.0530      0.050     -1.063      0.294      -0.154       0.047

"""

#Remove X2
X_opt = X[:,[0,1,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""Result: that means remove X3
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       2.744e+04   3008.359      9.122      0.000    2.14e+04    3.35e+04
x1          2.744e+04   3008.359      9.122      0.000    2.14e+04    3.35e+04
x2             0.8621      0.030     28.589      0.000       0.801       0.923
x3            -0.0530      0.049     -1.073      0.289      -0.152       0.046

"""

#remove X3
X_opt = X[:,[0,1,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#But my results are different from the tutorial.