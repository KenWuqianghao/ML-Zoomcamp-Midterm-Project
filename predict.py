import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from flask import Flask
from flask import request
from flask import jsonify

with open('model.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
    dv, rf = pickle.load(f_in)
f_in.close()

app = Flask('bike')
@app.route('/predict', methods=['POST'])

def predict():

    data = request.get_json()
    X = dv.transform([data])
    count = (np.expm1(rf.predict(X)))[0]

    return jsonify(count)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)