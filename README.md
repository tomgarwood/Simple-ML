# Simple ML for Australian Lawyer Salaries
A simple Python-based supervised regression machine learning model to predict Australian lawyer salaries based on PQE years, law firm tier, and location. PQE salary data gathered from various 2019 recruitment resources.

### Web App ###
Hosted on Heroku at: https://simple-ml.herokuapp.com/ (hosted on a free dyno, so it may take a second to spin up).

### Local Installation ###
1. git clone repository
2. cd Simple-ML
3. pip install -r requirements.txt
4. start Flask server in webapp.py

### Local Usage ####
The web app can be visited at http://0.0.0.0:5000/ once the Flask server is started. Enter location, PQE, and law firm tier to calculate salary. By default it runs from an included trained model ('model.joblib'), but can also be retrained against the included dataset ('/data/pqe_data_tier_city.csv'). 

### TODO ###
* Gather anonymised user data to expand dataset
