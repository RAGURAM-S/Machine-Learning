""" digits dataset"""
#from sklearn import datasets
#digits = datasets.load_digits()
#x = digits.data
#y = digits.target
#
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
#
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression()
#classifier.fit(x_train, y_train)
#
#prediction = classifier.predict(x_test)
#
#from sklearn import metrics
#mat = metrics.confusion_matrix(y_test, prediction)
#print("Accuracy of Logistic Regression Model in (%) ", metrics.accuracy_score(y_test, prediction)*100)


"""data.csv """
import pandas
data = pandas.read_csv("C:/Users/1024982/Desktop/data.csv")

x = data.iloc[:, 0:2].values
y = data.iloc[:, 2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, prediction)
matrix = metrics.confusion_matrix(y_test, prediction)

print('Accuracy in % : ', str(accuracy*100)+'%')

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plotter
import numpy as np
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
plotter.title('Logistic Regression (Training set)')
plotter.xlabel('X1')
plotter.ylabel('X2')
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
plotter.title('Logistic Regression (Test set)')
plotter.xlabel('X1')
plotter.ylabel('X2')
plotter.legend()
plotter.show()






"""geeks for geeks"""
#
#import csv 
#import numpy as np 
#import matplotlib.pyplot as plt 
#  
#  
#def loadCSV(filename): 
#    ''' 
#    function to load dataset 
#    '''
#    with open(filename,"r") as csvfile: 
#        lines = csv.reader(csvfile) 
#        dataset = list(lines) 
#        for i in range(len(dataset)): 
#            dataset[i] = [float(x) for x in dataset[i]]      
#    return np.array(dataset) 
#  
#  
#def normalize(X): 
#    ''' 
#    function to normalize feature matrix, X 
#    '''
#    mins = np.min(X, axis = 0) 
#    maxs = np.max(X, axis = 0) 
#    rng = maxs - mins 
#    norm_X = 1 - ((maxs - X)/rng) 
#    return norm_X 
#  
#  
#def logistic_func(beta, X): 
#    ''' 
#    logistic(sigmoid) function 
#    '''
#    return 1.0/(1 + np.exp(-np.dot(X, beta.T))) 
#  
#  
#def log_gradient(beta, X, y): 
#    ''' 
#    logistic gradient function 
#    '''
#    first_calc = logistic_func(beta, X) - y.reshape(X.shape[0], -1) 
#    final_calc = np.dot(first_calc.T, X) 
#    return final_calc 
#  
#  
#def cost_func(beta, X, y): 
#    ''' 
#    cost function, J 
#    '''
#    log_func_v = logistic_func(beta, X) 
#    y = np.squeeze(y) 
#    step1 = y * np.log(log_func_v) 
#    step2 = (1 - y) * np.log(1 - log_func_v) 
#    final = -step1 - step2 
#    return np.mean(final) 
#  
#  
#def grad_desc(X, y, beta, lr=.01, converge_change=.001): 
#    ''' 
#    gradient descent function 
#    '''
#    cost = cost_func(beta, X, y) 
#    change_cost = 1
#    num_iter = 1
#      
#    while(change_cost > converge_change): 
#        old_cost = cost 
#        beta = beta - (lr * log_gradient(beta, X, y)) 
#        cost = cost_func(beta, X, y) 
#        change_cost = old_cost - cost 
#        num_iter += 1
#      
#    return beta, num_iter  
#  
#  
#def pred_values(beta, X): 
#    ''' 
#    function to predict labels 
#    '''
#    pred_prob = logistic_func(beta, X) 
#    pred_value = np.where(pred_prob >= .5, 1, 0) 
#    return np.squeeze(pred_value) 
#  
#  
#def plot_reg(X, y, beta): 
#    ''' 
#    function to plot decision boundary 
#    '''
#    # labelled observations 
#    x_0 = X[np.where(y == 0.0)] 
#    x_1 = X[np.where(y == 1.0)] 
#      
#    # plotting points with diff color for diff label 
#    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0') 
#    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1') 
#      
#    # plotting decision boundary 
#    x1 = np.arange(0, 1, 0.1) 
#    x2 = -(beta[0,0] + beta[0,1]*x1)/beta[0,2] 
#    plt.plot(x1, x2, c='k', label='reg line') 
#  
#    plt.xlabel('x1') 
#    plt.ylabel('x2') 
#    plt.legend() 
#    plt.show() 
#      
#  
#      
#if __name__ == "__main__": 
#    # load the dataset 
#    dataset = loadCSV('C:/Users/1024982/Desktop/data.csv') 
#      
#    # normalizing feature matrix 
#    X = normalize(dataset[:, :-1]) 
#      
#    # stacking columns wth all ones in feature matrix 
#    X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X)) 
#  
#    # response vector 
#    y = dataset[:, -1] 
#  
#    # initial beta values 
#    beta = np.matrix(np.zeros(X.shape[1])) 
#  
#    # beta values after running gradient descent 
#    beta, num_iter = grad_desc(X, y, beta) 
#  
#    # estimated beta values and number of iterations 
#    print("Estimated regression coefficients:", beta) 
#    print("No. of iterations:", num_iter) 
#  
#    # predicted labels 
#    y_pred = pred_values(beta, X) 
#      
#    # number of correctly predicted labels 
#    print("Correctly predicted labels:", np.sum(y == y_pred)) 
#      
#    # plotting regression line 
#    plot_reg(X, y, beta) 