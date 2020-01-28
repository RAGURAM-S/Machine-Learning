from sklearn import datasets
dataset = datasets.load_boston()

x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
prediction = regressor.predict(x_test)

print('Coefficients : \n {}'.format(regressor.coef_ ))
print('Variance Score : \n {}'.format(regressor.score(x_train, y_train)))

import matplotlib.pyplot as plotter
#training dataset
plotter.style.use('fivethirtyeight')

plotter.scatter(regressor.predict(x_train), regressor.predict(x_train) - y_train, s = 10, color = 'green', label = 'Training data set')
#testing dataset
plotter.scatter(regressor.predict(x_test), regressor.predict(x_test) - y_test, s = 10, color = 'blue', label = 'Testing dataset')

plotter.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2, color = 'red')
plotter.legend(loc = 'lower left')
plotter.title('Residual Errors')
plotter.show()
