# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

#Technically if you do dataset.iloc[:,1], its the same, because we are just taking 1 column for X and Y
#But the result is a Vector, but for regression library, it requires the object to be
#Matrix instead of a vector, so we specify 1:2, even though its actually the same as 1
X_temp = dataset.iloc[:,1].values
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#No need to do splitting dataset because we only have 10 test data
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)"""

#No need to do feature scaling 

#Fit linear regression to the data set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#I think fit vs fit_transform. Fit_transform will return a new object

#Fit polynomial regression to the data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualizing the linear regression result
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression')
plt.show()

#VIsualizing the polynomial regression result
#Note, we dont directly use X_poly here, so that the code is more generic.
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Polynomial Regression')
plt.show()

#Changing to degree 3 and 4, we also make the graph more smooth by adding more resolution to X
X_grid = np.arange(min(X), max(X), 0.1)

#We need to re-shape X_grid into a matrix
X_grid = X_grid.reshape(len(X_grid),1)


poly_reg3 = PolynomialFeatures(degree=10)
X_poly3 = poly_reg3.fit_transform(X)

lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly3, y)

plt.scatter(X,y,color='red')
plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color='blue')
plt.title('Polynomial Regression')
plt.show()


#Making prediction with linear regression
lin_reg.predict([[6.5]])

#Making prediction with polynomial regression
lin_reg3.predict(poly_reg3.fit_transform([[6.5]]))