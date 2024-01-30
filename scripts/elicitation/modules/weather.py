import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from geopy.geocoders import Nominatim

def get_lat_long(city_name: str):
    """
    Get latitude and longitude of any city using OSM Nominatim API.

    Args:
      city_name: A string giving the name of the city.

    Examples:
      - get_lat_long('Paris')
      - get_lat_long('Paris, France', verbose=False)
      - get_lat_long('Paris, Texas')

    Returns:
      A tuple with (latitude, longitude)
    """
    geolocator = Nominatim(user_agent="dfki_dsa")
    location = geolocator.geocode(city_name)
    if location is None:
        raise ValueError(
            f"{city_name} not found on <https://nominatim.openstreetmap.org>")
    return {'address': location.address,
            'latitude': location.latitude,
            'longitude': location.longitude}

class WeatherAPI:
    def __init__(self, cities: list[str] = []):
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.cities = cities

    def get_weather(self, lat, lon,
                    start_date='2023-12-01', end_date='2023-12-31'):
        """
        Retrieve daily weather data from the OpenMeteo API.

        Args:
          lat: Latitude.
          lon: Longitude.
          start_date: Start date as string in YYYY-MM-DD
          end_date: End date as string in YYYY-MM-DD

        Returns:
          openmeteo API response object.
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ['temperature_2m_mean', 'precipitation_sum']
        }
        response = self.openmeteo.weather_api(url, params=params)
        return response

    def weather_to_df(self, response):
        """
        Convert openmeteo API response into DataFrame.

        Args:
          api_response: The output from get_weather().

        Returns:
          pd.DataFrame with columns 'date', 'temperature' and 'precipitation'.
        """
        daily = response.Daily()
        temperature = daily.Variables(0).ValuesAsNumpy()
        precipitation = daily.Variables(1).ValuesAsNumpy()
        date_range = pd.date_range(start=pd.to_datetime(daily.Time(), unit='s'),
                                   end=pd.to_datetime(
                                       daily.TimeEnd(), unit='s'),
                                   freq=pd.Timedelta(seconds=daily.Interval()),
                                   inclusive='left')
        return pd.DataFrame({'date': date_range,
                            'temperature': temperature,
                             'precipitation': precipitation})

    def get_weather_range(self, lat, lon,
                          start_year: int = 2010,
                          end_year: int = 2023,
                          month: int = 12):
        """
        Call get_weather for a series of months over different years.

        NB: only works for months with 31 days!
        """
        start_dates = [
            f'{year}-{month}-01' for year in range(start_year, end_year + 1)]
        end_dates = [
            f'{year}-{month}-31' for year in range(start_year, end_year + 1)]
        responses = [self.get_weather(lat, lon, start, end)[0] for (
            start, end) in zip(start_dates, end_dates)]
        return responses

    def weather_range_to_df(self, lst):
        return pd.concat([self.weather_to_df(x) for x in lst])

    def get_cities_weather(self):
        """
        Get historical daily rainfall and precipitation for a list of cities

        Args:
          cities: List of city names

        Returns:
          A pandas.DataFrame.
        """
        results = []
        for city in self.cities:
            loc = get_lat_long(city)
            lat, lon = loc['latitude'], loc['longitude']
            meteo_responses = self.get_weather_range(lat, lon)
            weather_df = self.weather_range_to_df(meteo_responses)
            weather_df['city'] = city
            weather_df['address'] = loc['address'] # just to check it's the right city!
            weather_df['latitude'], weather_df['longitude'] = lat, lon
            results.append(weather_df)
        self.data = pd.concat(results, ignore_index=True)

def example_weather_data():
    cities = ['Paris', 'Canberra', 'Kaiserslautern', 'Aoulef']
    weather = WeatherAPI(cities)
    weather.get_cities_weather()
    return weather.data

if __name__ == "__main__":
  example_data = example_weather_data()
  print(example_data.head())
