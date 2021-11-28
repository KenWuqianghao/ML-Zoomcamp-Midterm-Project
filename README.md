# ML-Zoomcamp-Midterm-Project

## Problem & Objective
In this project, I created a regression model predicitng the number of bike rented duirng a specific hour in London through information including temperature, wind speed, time, humidity, holiday, weather code and season. This model will produce a numerical output predicint how many bikes will be rented so the bike providers can distribute these bikes more wisely for maximum profit and usage. The dataset can be found [here](https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset).

## How to Use this Model in a Virtual Environment
1. Clone this repo to your local machine
2. cd into the directory and install dependencies using 
```python
pipenv install
```
3. Activate pipenv environment using
```bash
pipenv shell
```
4. Run predict.py
5. Run predict_test.py with whatever scenario you want to predict by changing the data variable

## How to Deploy to Docker
1. cd into the directory and create the docker image using
```docker
docker built -t bike-count .
```
2. Run the image by using
```docker
docker run -it -p 9696:9696 bike-count:latest
```
3. Run predict_test.py with whatever scenario you want to predict by changing the data variable
