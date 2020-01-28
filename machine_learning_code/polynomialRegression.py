import numpy
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plot

dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression/Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values

# linear regression model


linear_regressor = LinearRegression()

linear_regressor.fit(x, y)

linear_prediction = linear_regressor.predict(x)

#polynomial regression model

polynomial_regression = PolynomialFeatures(degree = 5)

x_polynomial = polynomial_regression.fit_transform(x)

linear_regressor_1 = LinearRegression()

linear_regressor_1.fit(x_polynomial, y)

polynomial_prediction = linear_regressor_1.predict(x_polynomial)

#linear regression model results
plot.scatter(x, y, color = 'blue')
plot.plot(x, linear_prediction, color = 'red')
plot.title('Linear Regression Model Results')
plot.xlabel('Position Levels in the Organization')
plot.ylabel('Salary')
plot.show()

x_grid = numpy.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plot.scatter(x, y, color = 'blue')
#plot.plot(x, polynomial_prediction, color = 'red')
plot.plot(x_grid, linear_regressor_1.predict(polynomial_regression.fit_transform(x_grid)), color = 'red')
plot.title('Polynomial Regression Model Results')
plot.xlabel('Position Levels in the Organization')
plot.ylabel('Salary')
plot.show()

#
#print(linear_regressor.predict([[6.5]]))
#print(linear_regressor_1.predict(numpy.array([[6.5]])))