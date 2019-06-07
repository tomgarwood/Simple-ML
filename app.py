import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset into pandas dataframe
data = pd.read_csv('pqe_data_syd.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# build regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# visualisation of training set
vistraining = plt
vistraining.scatter(X_train, y_train, color='red')
vistraining.plot(X_train, regressor.predict(X_train), color='blue')
vistraining.title('PQE Package (Training Set)')
vistraining.xlabel('PQE')
vistraining.ylabel('Package')
vistraining.show()

# visualisation of test set
vistest = plt
vistest.scatter(X_test, y_test, color='red')
vistest.plot(X_train, regressor.predict(X_train), color='blue')
vistest.title('PQE Package (Test Set)')
vistest.xlabel('PQE')
vistest.ylabel('Package')
vistest.show()

# PQE salary predictor
pqe = int(input('How many PQE are you: '))
y_pred = regressor.predict([[pqe]])
print(y_pred)





