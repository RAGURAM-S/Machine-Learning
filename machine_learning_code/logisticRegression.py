import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 14 - Logistic Regression/Logistic_Regression/Social_Network_Ads.csv")

x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

#prediction
predicted_results = classifier.predict(x_test)

#prediction metrics
from sklearn import metrics
matrix = metrics.confusion_matrix(y_test, predicted_results)
accuracy = metrics.accuracy_score(y_test, predicted_results)

#plotting the graph
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plotter
import numpy as np

#training set
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plotter.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('Logistic Regression (Training set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()


#test set
X_set_1, y_set_1 = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set_1[:, 0].min() - 1, stop = X_set_1[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set_1[:, 1].min() - 1, stop = X_set_1[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plotter.scatter(X_set_1[y_set_1 == j, 0], X_set_1[y_set_1 == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('Logistic Regression (Test set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()





