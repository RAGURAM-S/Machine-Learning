import numpy

x = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
y = numpy.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 

n = numpy.size(x)
mean_x = numpy.mean(x)
mean_y = numpy.mean(y)

#cross deviations
ss_xy = numpy.sum(y*x) - (n*mean_x*mean_y)
ss_xx = numpy.sum(x*x) - (n*mean_x*mean_x)

#regression coefficients
#y-intercept
b_1 = ss_xy/ss_xx
b_0 = mean_y - (b_1*mean_x)

print('Estimated Regression Coefficients are : \n b_0 = {} \n b_1 = {}'.format(b_0, b_1))

prediction = b_0 + b_1*x

import matplotlib.pyplot as plotter
plotter.scatter(x, y, color = 'blue')
plotter.plot(x, prediction, color = 'red')
plotter.title('Simple Linear Regression')
plotter.xlabel('X')
plotter.ylabel('Y')
plotter.show()

import numpy
import matplotlib.pyplot as plotter

def estimate_regression_coefficients(x, y):
    n = numpy.size(x)
    mean_x = numpy.mean(x)
    mean_y = numpy.mean(y)
    #cross deviation
    xy = numpy.sum(x*y) - n*(mean_x*mean_y)
    xx = numpy.sum(x**2) - n*(mean_x**2)
    #regression coefficients
    b_1 = xy/xx
    b_0 = mean_y - (b_1*mean_x)
    return (b_0, b_1)
    

def plot_graph(x, y, b):
    prediction = b[0] + b[1]*x
    plotter.scatter(x, y, color = 'blue')
    plotter.plot(x, prediction, color = 'red')
    plotter.title('Simple Linear Regression')
    plotter.xlabel('X')
    plotter.ylabel('Y')
    plotter.show()

def main():
    x = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = numpy.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    b = estimate_regression_coefficients(x, y)
    print("Estimated Regression Coefficients are : \n b_0 = {} \n b_1 = {}".format(b[0], b[1]))
    plot_graph(x,y,b)

if __name__ == "__main__":
    main()