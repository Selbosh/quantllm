from pathlib import Path
import pandas as pd
from modules.weather import WeatherAPI

folder = Path(__file__).parents[2] / 'data/cities'
folder.mkdir(parents=True, exist_ok=True)

to_save = folder / 'city_names.txt'
if to_save.exists():
    city_names = open(to_save, 'r').readlines()
else:
    raise FileNotFoundError('Please supply a list of city names in city_names.txt')

weather = WeatherAPI(city_names)
weather.get_cities_weather()
dataset = weather.data
dataset.to_csv(folder / 'cities.csv', index=False)
