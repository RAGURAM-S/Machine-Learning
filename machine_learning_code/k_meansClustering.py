import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/K_Means/Mall_Customers.csv")

x = dataset.iloc[:, [3,4]].values

# determining the optiamal number of clusters using the elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    cluster = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    cluster.fit(x)
    wcss.append(cluster.inertia_)
    

import matplotlib.pyplot as plotter
plotter.plot(range(1,11), wcss)
plotter.title('The Elbow Method for determining the optimal number of clusters')
plotter.xlabel('The Number of clusters')
plotter.ylabel('WCSS')
plotter.show()

#clustering 
cluster = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
prediction = cluster.fit_predict(x)

#graph

#plotting the clusters
plotter.scatter(x[prediction == 0,0], x[prediction == 0,1], s = 100, color = 'cyan', label = 'careful')
plotter.scatter(x[prediction == 1,0], x[prediction == 1,1], s = 100, color = 'blue', label = 'standard')
plotter.scatter(x[prediction == 2,0], x[prediction == 2,1], s = 100, color = 'green', label = 'target')
plotter.scatter(x[prediction == 3,0], x[prediction == 3,1], s = 100, color = 'red', label = 'reckless')
plotter.scatter(x[prediction == 4,0], x[prediction == 4,1], s = 100, color = 'magenta', label = 'sensible')

#plotting the centroids
plotter.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], s = 300, color = 'yellow')
plotter.title('Clustering of Customers - K-Means Algorithm')
plotter.xlabel('Annual Income in k$')
plotter.ylabel('Spending Score')
plotter.legend()
plotter.show()
