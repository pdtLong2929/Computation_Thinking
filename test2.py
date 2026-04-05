import requests

API_URL = "http://localhost:8000"
response = requests.get(API_URL)
print(response.json())

###################################################

API_URL = "http://localhost:8000/health"
response = requests.get(API_URL)
print(response.json())

########################################################

API_URL = "http://localhost:8000/predict"
data = {
    "text": "Hello",
    "max_length": 50,
    "min_length": 20
}

response = requests.post(API_URL, json=data)
print(response.json())