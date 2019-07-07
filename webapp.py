import os
from flask import Flask, render_template, request
from main import regressor

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def userform():
    loc = request.form['loc']
    pqe = float(request.form['pqe'])
    tier = request.form['tier']
    salary = '${:0,.0f}'.format(regressor(loc, pqe, tier))
    return render_template('result.html', salary=salary, loc=loc, pqe=pqe, tier=tier)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')
