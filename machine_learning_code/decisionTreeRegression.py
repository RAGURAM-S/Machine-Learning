import pandas
import numpy
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression/Decision_Tree_Regression/Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)
a = numpy.array([6.5])
a = a.reshape(-1,1)
prediction = regressor.predict(a)

x_grid = numpy.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(-1,1)

import matplotlib.pyplot as plotter
plotter.scatter(x, y, color = 'blue')
plotter.plot(x_grid, regressor.predict(x_grid), color = 'red')
plotter.title('Decision Tree Regression  Model Results')
plotter.xlabel('Position in the Organization')
plotter.ylabel('Salary')
plotter.show()