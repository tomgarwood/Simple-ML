import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load

# import dataset into a pandas dataframe
"""
data = pd.read_csv('data/pqe_data_tier_city.csv')
data['tier'] = data['tier'].replace({1: 'top', 2: 'mid', 3: 'small'})
data = pd.get_dummies(data, columns=['tier'])
data['location'] = data['location'].replace({1: 'syd', 2: 'melb', 3: 'bris', 4: 'per'})
data = pd.get_dummies(data, columns=['location'])
"""

# create test/train data split
"""
X = data.drop('package', axis=1)
y = data[['package']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
"""

# train machine learning regression model
"""
regressor = LinearRegression()
regressor.fit(X_train, y_train)
dump(regressor, 'model/model.joblib')  # dump trained model for future use
"""

# test tools, uncomment to review
"""
from sklearn.metrics import mean_squared_error
import math
for i, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regressor.coef_[0][i]))
intercept = regressor.intercept_[0]
print("The model incercept is {}".format(intercept))
y_predit = regressor.predict(X_test)
regressor_mse = mean_squared_error(y_predit, y_test)
print(math.sqrt(regressor_mse))
print(regressor.score(X_test, y_test))
"""

# prediction output from regressor model
def regressor(loc, pqe, tier):
    syd = float(1) if loc == 'syd' else float(0)
    mel = float(1) if loc == 'mel' else float(0)
    bris = float(1) if loc == 'bris' else float(0)
    per = float(1) if loc == 'per' else float(0)
    top = float(1) if tier == 'top' else float(0)
    mid = float(1) if tier == 'mid' else float(0)
    small = float(1) if tier == 'small' else float(0)
    regressor = load('model/model.joblib')  # loading pre-trained model
    result = regressor.predict([[pqe, mid, small, top, bris, mel, per, syd]])
    return result[0][0]
