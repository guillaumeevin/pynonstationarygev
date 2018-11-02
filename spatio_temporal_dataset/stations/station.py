import pandas as pd
import numpy as np


class Station(object):
    def __init__(self, name: str, annual_maxima: pd.Series, longitude=np.nan, latitude=np.nan, altitude=np.nan):
        self.annual_maxima = annual_maxima
        self.year_of_study = list(annual_maxima.index)
        self.name = name
        self.altitude = altitude
        self.latitude = latitude
        self.longitude = longitude
        self.distance = {}


def load_stations_from_dataframe(df):
    return [Station(name=i, annual_maxima=row) for i, row in df.iterrows()]

def load_station_from_two_dataframe(df, location_df):
    pass


if __name__ == '__main__':
    df = pd.DataFrame(1, index=['station1', 'station2'], columns=['200' + str(i) for i in range(9)])
    stations = load_stations_from_dataframe(df)
    station = stations[0]
    print(station.name)
    print(station.annual_maxima)
    print(station.year_of_study)
