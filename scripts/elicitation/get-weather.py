from pathlib import Path
import pandas as pd
from modules.weather import WeatherAPI

folder = Path(__file__).parents[2] / 'data/cities'
folder.mkdir(parents=True, exist_ok=True)

# Download weather data (temperature and precipitation in December) for cities:
city_names_file = folder / 'city_names.txt'
if city_names_file.exists():
    city_names = open(city_names_file, 'r').readlines()
else:
    raise FileNotFoundError('Please supply a list of city names in city_names.txt')

weather = WeatherAPI(city_names)
weather.get_cities_weather()
dataset = weather.data
dataset.to_csv(folder / 'cities.csv', index=False)

# Commenting out for now because the dataset doesn't seem to work:
# ----------------------------------------------------------------
# Download mean BMI estimates by country from the World Health Organization:
# https://www.who.int/data/gho/data/indicators/indicator-details/GHO/mean-bmi-(kg-m-)-(age-standardized-estimate)
# who_gho_file = folder / 'who_bmi.csv'
# if not who_gho_file.exists():
#   print('Please download the following dataset from the WHO Global Health Observatory\n'
#         'https://www.who.int/data/gho/data/indicators/indicator-details/GHO/mean-bmi-(kg-m-)-(age-standardized-estimate)\n'
#         'Then save it in path: ' + str(who_gho_file))
