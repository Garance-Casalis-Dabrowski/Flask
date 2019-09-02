import requests
import json

url = 'http://localhost:5000/prediction'
r = requests.post(url,json={'features':1.5})
print(r.json())
