import requests

url = 'http://localhost:9696/predict'
data = {'timestamp':'9:00', 't1':15, ' t2': 15, 'hum':100, 'wind_speed':4.0, 'weather_code': 2}
response = requests.post(url, json=data).json()

print(response)