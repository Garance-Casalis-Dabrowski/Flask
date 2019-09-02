import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import requests
import json


model = LogisticRegression(random_state=0)
#model.fit(data[['size']], data['is_malignant'])
#model.score(data[['size']], data['is_malignant'])
to_predict = np.array([[0.414], [0.001], [1.1], [2000]])
# print(to_predict.shape)
model.predict(to_predict)

X = data[['size']]
Y = data['is_malignant']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

#print(X_train.shape)
# problemes de dimensions X_train et Y_train ont des dimensions opposées 
model = LogisticRegression(solver='liblinear')
#model.fit(X_train,Y_train)
#print(model.score(X_train, Y_train))
#print(model.score(X_test, Y_test))

# filename = 'model.pkl'
# with open(filename, 'wb') as f: 
#     pickle.dump(model, f)
# print(model.predict([[1.8]]))


# save the model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
 
# # some time later...
print(model.predict([[1.8]]))
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

def load_model():
	model = pickle.load(open('../models/finalized_model.sav','rb'))
    


def predict(data):
	model.predict()
    
