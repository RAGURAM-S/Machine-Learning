import pandas
dataset = pandas.read_csv("C:\ml\Machine Learning A-Z Template Folder\Part 10 - Model Selection & Boosting\Section 48 - Model Selection\Model_Selection\Social_Network_Ads.csv")

x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

#Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracy_vector = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
mean_accuracy = accuracy_vector.mean()
standard_deviation = accuracy_vector.std()

prediction = classifier.predict(x_test)

#performance evaluation
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, prediction)
accuracy_score = metrics.accuracy_score(y_test, prediction)

#visualization of results using graph
import numpy as np
import matplotlib.pyplot as plotter
from matplotlib.colors import ListedColormap


#training set
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plotter.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('SVM (Training set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()
#test set
x_set_1, y_set_1 = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set_1[:, 0].min() - 1, stop = x_set_1[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set_1[:, 1].min() - 1, stop = x_set_1[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plotter.scatter(x_set_1[y_set_1 == j, 0], x_set_1[y_set_1 == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('SVM (Test set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()

