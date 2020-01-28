import numpy as np
from matplotlib import pyplot as plot
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as stats

dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/50_Startups.csv")

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:,-1].values


# Categorical data - Encoding independent variables
label_encoder_x = LabelEncoder()

#
one_hot_encoder = OneHotEncoder(categorical_features = [3])

x[:, 3] = label_encoder_x.fit_transform(x[:,3])

x = one_hot_encoder.fit_transform(x).toarray()

# Avoids Dummy Variable trap
x = x[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_prediction = regressor.predict(x_test)


# Backward Elimination algorithm

x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis =1)

x_optimal = x[:, [0, 1, 2, 3, 4, 5]]

regressor_OLS = stats.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

x_optimal = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = stats.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

x_optimal = x[:, [0, 3, 4, 5]]
regressor_OLS = stats.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

x_optimal = x[:, [0, 3, 5]]
regressor_OLS = stats.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

x_optimal = x[:, [0, 3]]
regressor_OLS = stats.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

