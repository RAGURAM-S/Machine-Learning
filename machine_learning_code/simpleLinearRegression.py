import pandas
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pandas.read_csv('C:\ml\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Simple_Linear_Regression/Salary_Data.csv')

#x- experience
x = dataset.iloc[:, 0].values
#salary
y = dataset.iloc[:, 1].values

#converting x into a 2D numpy array
x = x.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plot.scatter(x_train, y_train, color='blue')
plot.plot(x_train, regressor.predict(x_train),color = 'red')
plot.title('Salary vs Expperience(Training Set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

plot.scatter(x_test, y_test, color = 'blue')
plot.plot(x_train, regressor.predict(x_train),color = 'red')
plot.title('Salary vs Experience(Test Set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()
