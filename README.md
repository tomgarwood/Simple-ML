# Simple ML for Australian Lawyer Salaries
A simple Python regression-based machine learning model to predict Australian lawyer salaries based on PQE years, law firm tier, and location. PQE data gathered from various 2019 recruitment resources.

### Installation ###
1. git clone repository
2. cd Simple-ML
3. pip install -r requirements.txt

### Usage ####
The script can be run from the included trained model ('model.joblib') or retrained against the included dataset ('/data/pqe_data_tier_city.csv'). Run 'main.py' and the script will prompt for user input on PQE, location and law firm tier before generating the estimated salary.

### TODO ###
* Turn into Flask webapp
    * Gather anonymised user data to expand dataset
