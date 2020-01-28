import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/K_Means/Mall_Customers.csv")

x = dataset.iloc[:, [3, 4]].values

# Dendrograms to determine the optimal number of clusters
import matplotlib.pyplot as plotter
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plotter.title('Dendrogram')
plotter.xlabel('Customers')
plotter.ylabel('Euclidean Distance')
plotter.show()

#clustering 
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
prediction = cluster.fit_predict(x)

#graph
#plotting the clusters
plotter.scatter(x[prediction == 0,0], x[prediction == 0,1], s = 100, color = 'cyan', label = 'careful')
plotter.scatter(x[prediction == 1,0], x[prediction == 1,1], s = 100, color = 'blue', label = 'standard')
plotter.scatter(x[prediction == 2,0], x[prediction == 2,1], s = 100, color = 'green', label = 'target')
plotter.scatter(x[prediction == 3,0], x[prediction == 3,1], s = 100, color = 'red', label = 'reckless')
plotter.scatter(x[prediction == 4,0], x[prediction == 4,1], s = 100, color = 'magenta', label = 'sensible')
plotter.title('Clustering of Customers - Hierarchical Clustering')
plotter.xlabel('Annual Income in K$')
plotter.ylabel('Spending Score')
plotter.legend()
plotter.show()