import numpy
import pandas
import matplotlib.pyplot as plotter
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/SVR/Position_Salaries.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

y = y.reshape(-1, 1)

"""splitting the dataset into training data set and testing data set"""

"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_traIn, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

#feature scaling
scale_x = StandardScaler()
scale_y = StandardScaler()
x = scale_x.fit_transform(x)
y = scale_y.fit_transform(y)

"""SVR regression"""

svr_regressor = SVR(kernel = 'rbf')

svr_regressor.fit(x, y)

prediction = svr_regressor.predict(x)


y_prediction = scale_y.inverse_transform(svr_regressor.predict(scale_x.transform(numpy.array([[6.5]]))))

"""smooth curve in the graph"""

x_grid = numpy.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

""" plotting the graph"""

plotter.scatter(x, y, color = 'blue')
plotter.plot(x_grid, svr_regressor.predict(x_grid), color = 'red')
plotter.title('SVR Regression Model Results')
plotter.xlabel('Position Levels in the organization')
plotter.ylabel('Salary')
plotter.show()

