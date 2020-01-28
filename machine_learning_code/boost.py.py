import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 49 - XGBoost/XGBoost/Churn_Modelling.csv")

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder_geography = LabelEncoder()
label_encoder_gender = LabelEncoder()

x[:, 1] = label_encoder_geography.fit_transform(x[:, 1])
x[:, 2] = label_encoder_gender.fit_transform(x[:, 2])

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(categorical_features = [1])
x = one_hot_encoder.fit_transform(x).toarray()


#removing the dummy variable trap
x = x[:, 1:]


#splitting the training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#importing the xgboost algorithm
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

#predictions
prediction = classifier.predict(x_test)


#evaluation of performance
from sklearn import metrics
accuracy_score = metrics.accuracy_score(y_test, prediction)
confusion_matrix = metrics.confusion_matrix(y_test, prediction)


#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracy_vector = cross_val_score(estimator = classifier, 
                                  X = x_train, 
                                  y = y_train, 
                                  cv = 10,
                                  n_jobs = -1)
mean_accuracy = accuracy_vector.mean()
standard_deviation = accuracy_vector.std()