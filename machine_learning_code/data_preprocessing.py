import numpy as np
import pandas
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset= pandas.read_csv('C:/ml/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data_Preprocessing/Data.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

imputer= Imputer(missing_values= np.nan, strategy= 'mean')
imputer= imputer.fit(x[:, 1:3])
x[:, 1:3]= imputer.transform(x[:, 1:3])


labelEncoder_x= LabelEncoder()
labelEncoder_y= LabelEncoder()


x[:, 0]= labelEncoder_x.fit_transform(x[:, 0])


oneHotEncoder= OneHotEncoder(categorical_features=[0])

x=oneHotEncoder.fit_transform(x).toarray()
y= labelEncoder_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

scale_x = StandardScaler()

x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)

#print(x)
#print(y)
 
