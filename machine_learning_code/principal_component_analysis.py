import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/PCA/Wine.csv")

x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Applying Pricipal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

#prediction
prediction = classifier.predict(x_test)

#performance evaluation using metrics
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, prediction)
accuracy = metrics.accuracy_score(y_test, prediction)

#visualization of results
import numpy as np
import matplotlib.pyplot as plotter
from matplotlib.colors import ListedColormap

#import warnings
#warnings.filterwarnings('ignore')

#training set
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plotter.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plotter.title('Logistic Regression (Training set)')
plotter.xlabel('Principal Component 1')
plotter.ylabel('Principal Component 2')
plotter.legend()
plotter.show()


#test set
X_set_1, y_set_1 = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set_1[:, 0].min() - 1, stop = X_set_1[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set_1[:, 1].min() - 1, stop = X_set_1[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plotter.scatter(X_set_1[y_set_1 == j, 0], X_set_1[y_set_1 == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plotter.title('Logistic Regression (Test set)')
plotter.xlabel('Principal Component 1')
plotter.ylabel('Principal Component 2')
plotter.legend()
plotter.show()





