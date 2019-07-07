# Simple ML for Australian Lawyer Salaries
A simple Python-based supervised regression machine learning model to predict Australian lawyer salaries based on PQE years, law firm tier, and location. PQE data gathered from various 2019 recruitment resources.

### Web App ###
Hosted on Heroku at: https://simple-ml.herokuapp.com/

### Local Installation ###
1. git clone repository
2. cd Simple-ML
3. pip install -r requirements.txt
4. start Flask server in webapp.py

### Local Usage ####
The script can be run from the included trained model ('model.joblib') or retrained against the included dataset ('/data/pqe_data_tier_city.csv'). Run 'main.py' and the script will prompt for user input on PQE, location and law firm tier before generating the estimated salary. Hosted version available here: Hosted on Heroku at: https://simple-ml.herokuapp.com/ 

### TODO ###
* Gather anonymised user data to expand dataset
