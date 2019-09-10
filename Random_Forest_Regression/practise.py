# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values #HT: y needs to be a matrix, otherwise feature scaling fails

#Fitting the random forest regression to the dataset. We select 10 trees**
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators  = 100, random_state = 0)
regressor.fit(X,y)

#Predicting a new result
y_pred = regressor.predict([[6.5]])


# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
#As the random forest is non-continuous, we need to increase the resolution for our plot
#We are getting more steps.. But doesnt mean that if we add more trees, doesnt mean 
# that we will always get more steps.. It all depends on entropy, so eventually it will converge to a number of steps..
#if we put 300 trees , we get around 160K that is closest to the actual value.
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

