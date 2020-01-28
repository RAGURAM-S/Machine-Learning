import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Artificial_Neural_Networks/Churn_Modeling.csv")

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


#encoding the categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder_geography = LabelEncoder()
label_encoder_gender = LabelEncoder()
x[:, 1] = label_encoder_geography.fit_transform(x[:, 1])
x[:, 2] = label_encoder_gender.fit_transform(x[:, 2])

one_hot_encoder = OneHotEncoder(categorical_features = [1])
x = one_hot_encoder.fit_transform(x).toarray()


#avoiding dummy variable trap
x = x[:, 1:]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

""" Artificial Neural Networks"""


# importing the relevant packages

from keras.models import Sequential
from keras.layers import Dense


# initializing the Artificial Neural Netowork
classifier = Sequential()


# adding the inpyt layer and the first hidden layer
"""activation function for the hidden layers is generally set as the rectifier function - relu and that for the output layer is set to sigmoid function"""
"""in case the output layer contains more classes/ categories the softmax function is deployed as the activation function"""

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))


#adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


#adding the third hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


#output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#compiling the artificial neural network
"""efficient stochastic gradient descent algorithm - adam"""

"""binary_crossentropy is the logarithmic loss function if the dependent variable is a binary outcome
 and if the number of dependent variables are greater than 2 then the loss function is categorical_crossentropy"""

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#fitting the artificial neural network to the training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 200)


#prediction of the results for the test set
prediction = classifier.predict(x_test)
prediction = (prediction > 0.5)


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, prediction)
accuracy = metrics.accuracy_score(y_test, prediction) 