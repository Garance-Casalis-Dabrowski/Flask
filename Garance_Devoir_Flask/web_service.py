from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle 
import json 
#from model import * 

app = Flask(__name__)
model = pickle.load(open('models/finalized_model.sav','rb'))

@app.route('/prediction', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = int(model.predict([[data['features']]])[0])
    output = prediction
    return jsonify(output)

if __name__ == '__main__':
    modelfile = 'models/finalized_model.sav'
    model = pickle.load(open(modelfile, 'rb'))
    app.run(port=5000, debug=True)
	
