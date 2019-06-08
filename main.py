import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import math

# import dataset into a pandas dataframe
data = pd.read_csv('data/pqe_data_tier_city.csv')
data['tier'] = data['tier'].replace({1: 'top', 2: 'mid', 3: 'small'})
data = pd.get_dummies(data, columns=['tier'])
data['location'] = data['location'].replace({1: 'syd', 2: 'melb', 3: 'bris', 4: 'per'})
data = pd.get_dummies(data, columns=['location'])
print(data.head())

# create test/train data split
X = data.drop('package', axis=1)
y = data[['package']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# train machine learning regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
dump(regressor, 'model.joblib')  # dump trained model for future use

# uncomment to load trained model
# regressor = load('model.joblib')

# test tools, uncomment to review
"""
for i, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regressor.coef_[0][i]))
intercept = regressor.intercept_[0]
print("The intercept for this model is {}".format(intercept))
y_predit = regressor.predict(X_test)
regressor_mse = mean_squared_error(y_predit, y_test)
print(math.sqrt(regressor_mse))
print(regressor.score(X_test, y_test))
"""

print(regressor.predict([[3, 0, 0, 1, 0, 0, 0, 1]]))

