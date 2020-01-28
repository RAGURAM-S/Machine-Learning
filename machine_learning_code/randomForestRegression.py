import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression/Decision_Tree_Regression/Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#n_estimators is the number of decision trees taken into account
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(x, y)

import numpy
a = numpy.array([6.5])
a = a.reshape(len(a), 1)

prediction = regressor.predict(a)

x_grid = numpy.arange(min(x),max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid),1)

import matplotlib.pyplot as plotter
plotter.scatter(x, y, color = 'blue')
plotter.plot(x_grid, regressor.predict(x_grid), color = 'red')
plotter.title('Random Forest Regression Model')
plotter.xlabel('Position in the Organization')
plotter.ylabel('Salary')
plotter.show()