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
    "text": "The Apollo program was the third United States human spaceflight program carried out by NASA, which succeeded in landing the first humans on the Moon from 1968 to 1972. It was first conceived in 1960 during President Dwight D. Eisenhower's administration.",
    "max_length": 50,
    "min_length": 20
}
response = requests.post(API_URL, json=data)
print(response.json())