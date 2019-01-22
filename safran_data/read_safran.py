import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import os.path as op

import pandas as pd
from netCDF4 import Dataset

from extreme_estimator.gev.fit_gev import gev_mle_fit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from utils import get_full_path
from collections import OrderedDict
from datetime import datetime, timedelta


class Safran(object):

    def __init__(self):
        self.year_to_dataset = OrderedDict()
        print(self.safran_full_path)
        nc_files = [(self.str_to_year(file), file) for file in os.listdir(self.safran_full_path) if file.endswith('.nc')]
        for year, nc_file in sorted(nc_files, key=lambda t: t[0]):
            self.year_to_dataset[year] = Dataset(op.join(self.safran_full_path, nc_file))
        #
        # Map each index to the corresponding massif
        print(list(self.year_to_dataset.keys()))
        massifs_str = self.year_to_dataset[1958].massifsList.split('/')
        self.int_to_massif = {j: Massif.from_str(massif_str) for j, massif_str in enumerate(massifs_str)}
        safran_massif_names = set([massif.name for massif in self.int_to_massif.values()])

        # # Map each datetime to a snowfall amount in kg/m2
        self.datetime_to_snowfall_amount = OrderedDict()
        self.year_to_snowfall_amount = {}
        for year, dataset in self.year_to_dataset.items():
            # Starting date
            start_datetime = self.new_year_day(year)
            start_times = np.array(dataset.variables['time'])[:-1]
            snowfall_rates = np.array(dataset.variables['Snowf'])
            mean_snowfall_rates = 0.5 * (snowfall_rates[:-1] + snowfall_rates[1:])
            self.year_to_snowfall_amount[year] = []
            for start_time, mean_snowrate in zip(start_times, mean_snowfall_rates):
                # time is in seconds, snowfall is in kg/m2/s
                delta_time = timedelta(seconds=start_time)
                current_datetime = start_datetime + delta_time
                snowfall_in_one_hour = 60 * 60 * mean_snowrate
                # TODO: for the datetime, I should put the middle of the interval, isntead of the start
                self.datetime_to_snowfall_amount[current_datetime] = snowfall_in_one_hour
                self.year_to_snowfall_amount[year].append(snowfall_in_one_hour)

        # Extract the maxima, and fit a GEV on each massif
        massif_to_maxima = {massif_name: [] for massif_name in safran_massif_names}
        for year, snowfall_amount in self.year_to_snowfall_amount.items():
            snowfall_amount = np.array(snowfall_amount)
            print(snowfall_amount.shape)
            max_snowfall_amount = snowfall_amount.max(axis=0)
            # todo: take the daily maxima

            print(max_snowfall_amount.shape)
            for i, massif in self.int_to_massif.items():
                massif_to_maxima[massif.name].append(max_snowfall_amount[i])
        print(massif_to_maxima)

        # Fit a gev n each massif
        # todo: find another for the gev fit
        massif_to_gev_mle = {massif: gev_mle_fit(np.array(massif_to_maxima[massif])) for massif in safran_massif_names}

        # Visualize the massif
        # todo: adapt the color depending on the value (create 3 plots, one for each gev aprameters) REFACTOR THE CODE
        self.visualize(safran_massif_names)

    def visualize(self, safran_massif_names):
        # Map each index to the correspond centroid
        df_centroid = pd.read_csv(op.join(self.map_full_path, 'coordonnees_massifs_alpes.csv'))
        for coord_column in [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y]:
            df_centroid.loc[:, coord_column] = df_centroid[coord_column].str.replace(',', '.').astype(float)
        coordinate_massif_names = set(df_centroid['NOM'])
        # Assert that the massif names are the same between SAFRAN and the coordinates
        assert not set(safran_massif_names).symmetric_difference(coordinate_massif_names)
        # todo: the coordinate are probably in Lambert extended
        # Build coordinate object
        print(df_centroid.dtypes)
        coordinate = AbstractSpatialCoordinates.from_df(df_centroid)
        # Load the
        df_massif = pd.read_csv(op.join(self.map_full_path, 'massifsalpes.csv'))
        coord_tuples = [(row_massif['idx'], row_massif[AbstractCoordinates.COORDINATE_X],
                         row_massif[AbstractCoordinates.COORDINATE_Y])
                        for _, row_massif in df_massif.iterrows()]
        for massif_idx in set([tuple[0] for tuple in coord_tuples]):
            l = [coords for idx, *coords in coord_tuples if idx == massif_idx]
            l = list(zip(*l))
            plt.plot(*l, color='black')
            plt.fill(*l)
            coordinate.visualization_2D()


    def new_year_day(self, year):
        return datetime(year=year, month=8, day=1, hour=6)

    def year_to_daily_maxima(self, nb_days=1):
        deltatime = timedelta(days=nb_days)
        start_time, *_, end_time = self.datetime_to_snowfall_amount.keys()
        total_snowfall = np.zeros(shape=(len(self.datetime_to_snowfall_amount), len(self.int_to_massif)))
        print(total_snowfall.shape)
        # for initial_time in self.datetime_to_snowfall_amount.keys():
        #     final_time = initial_time + deltatime
        #     new_year = self.new_year_day(initial_time.year)
        #     # two dates should correspond to the same year
        #     if not ((final_time > new_year) and (initial_time < new_year)):
        #
        # print(start_time)
        # print(end_time)

    @staticmethod
    def str_to_year(s):
        return int(s.split('_')[1][:4])

    @property
    def full_path(self) -> str:
        return get_full_path(relative_path=self.relative_path)

    @property
    def relative_path(self) -> str:
        return r'local/spatio_temporal_datasets'

    @property
    def safran_altitude(self) -> int:
        raise NotImplementedError

    @property
    def safran_full_path(self) -> str:
        return op.join(self.full_path, 'safran-crocus_{}'.format(self.safran_altitude), 'Safran')

    @property
    def map_full_path(self):
        return op.join(self.full_path, 'map')


class Safran1800(Safran):

    @property
    def safran_altitude(self) -> int:
        return 1800

class Safran2400(Safran):

    @property
    def safran_altitude(self) -> int:
        return 2400


class Massif(object):

    def __init__(self, name: str, id: int, lat: float, lon: float) -> None:
        self.lon = lon
        self.lat = lat
        self.id = id
        self.name = name

    @classmethod
    def from_str(cls, s: str):
        name, id, lat, lon = s.split(',')
        return cls(name.strip(), int(id), float(lat), float(lon))


if __name__ == '__main__':
    safran_object = Safran1800()
    # print(safran_object.year_to_daily_maxima(nb_days=3))
