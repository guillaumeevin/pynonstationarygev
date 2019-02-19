import os
import os.path as op
from collections import OrderedDict
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable
from experiment.meteo_france_SCM_study.massif import safran_massif_names_from_datasets
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
from utils import get_full_path, cached_property


class AbstractStudy(object):
    ALTITUDES = [1800, 2400]

    def __init__(self, variable_class, altitude=1800):
        assert altitude in self.ALTITUDES
        self.altitude = altitude
        self.model_name = None
        self.variable_class = variable_class

    def write_to_file(self, df):
        if not op.exists(self.result_full_path):
            os.makedirs(self.result_full_path, exist_ok=True)
        df.to_csv(op.join(self.result_full_path, 'merged_array_{}_altitude.csv'.format(self.altitude)))

    """ Data """

    @property
    def df_all_snowfall_concatenated(self) -> pd.DataFrame:
        df_list = [pd.DataFrame(snowfall, columns=self.safran_massif_names) for snowfall in
                   self.year_to_daily_time_serie.values()]
        df_concatenated = pd.concat(df_list)
        return df_concatenated

    @property
    def observations_annual_maxima(self) -> AnnualMaxima:
        return AnnualMaxima(df_maxima_gev=pd.DataFrame(self.year_to_annual_maxima, index=self.safran_massif_names))

    """ Load some attributes only once """

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_maxima = OrderedDict()
        for year, time_serie in self.year_to_daily_time_serie.items():
            year_to_annual_maxima[year] = time_serie.max(axis=0)
        return year_to_annual_maxima

    @cached_property
    def year_to_dataset_ordered_dict(self) -> OrderedDict:
        # Map each year to the correspond netCDF4 Dataset
        year_to_dataset = OrderedDict()
        nc_files = [(int(f.split('_')[1][:4]), f) for f in os.listdir(self.safran_full_path) if f.endswith('.nc')]
        for year, nc_file in sorted(nc_files, key=lambda t: t[0]):
            year_to_dataset[year] = Dataset(op.join(self.safran_full_path, nc_file))
        return year_to_dataset

    @cached_property
    def year_to_daily_time_serie(self) -> OrderedDict:
        # Map each year to a matrix of size 365-nb_days_consecutive+1 x nb_massifs
        year_to_variable = {year: self.instantiate_variable_object(dataset) for year, dataset in
                            self.year_to_dataset_ordered_dict.items()}
        year_to_daily_time_serie = OrderedDict()
        for year in self.year_to_dataset_ordered_dict.keys():
            year_to_daily_time_serie[year] = year_to_variable[year].daily_time_serie
        return year_to_daily_time_serie

    def instantiate_variable_object(self, dataset) -> AbstractVariable:
        return self.variable_class(dataset)

    ##########

    @property
    def safran_massif_names(self) -> List[str]:
        # Load the names of the massif as defined by SAFRAN
        return safran_massif_names_from_datasets(list(self.year_to_dataset_ordered_dict.values()))

    @property
    def safran_massif_id_to_massif_name(self) -> Dict[int, str]:
        return {massif_id: massif_name for massif_id, massif_name in enumerate(self.safran_massif_names)}

    @cached_property
    def massifs_coordinates(self) -> AbstractSpatialCoordinates:
        # Coordinate object that represents the massif coordinates in Lambert extended
        df_centroid = self.load_df_centroid()
        for coord_column in [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y]:
            df_centroid.loc[:, coord_column] = df_centroid[coord_column].str.replace(',', '.').astype(float)
        # Build coordinate object from df_centroid
        return AbstractSpatialCoordinates.from_df(df_centroid)

    def load_df_centroid(self) -> pd.DataFrame:
        df_centroid = pd.read_csv(op.join(self.map_full_path, 'coordonnees_massifs_alpes.csv'))
        # Assert that the massif names are the same between SAFRAN and the coordinate file
        assert not set(self.safran_massif_names).symmetric_difference(set(df_centroid['NOM']))
        df_centroid.set_index('NOM', inplace=True)
        df_centroid = df_centroid.loc[self.safran_massif_names]
        return df_centroid

    @property
    def coordinate_id_to_massif_name(self) -> Dict[int, str]:
        df_centroid = self.load_df_centroid()
        return dict(zip(df_centroid['id'], df_centroid.index))

    """ Visualization methods """

    def visualize(self, ax=None, massif_name_to_fill_kwargs=None, show=True, fill=True):
        print("here")
        if ax is None:
            ax = plt.gca()
        df_massif = pd.read_csv(op.join(self.map_full_path, 'massifsalpes.csv'))
        coord_tuples = [(row_massif['idx'], row_massif[AbstractCoordinates.COORDINATE_X],
                         row_massif[AbstractCoordinates.COORDINATE_Y])
                        for _, row_massif in df_massif.iterrows()]

        for coordinate_id in set([tuple[0] for tuple in coord_tuples]):
            l = [coords for idx, *coords in coord_tuples if idx == coordinate_id]
            l = list(zip(*l))
            ax.plot(*l, color='black')
            if fill:
                massif_name = self.coordinate_id_to_massif_name[coordinate_id]
                fill_kwargs = massif_name_to_fill_kwargs[massif_name] if massif_name_to_fill_kwargs is not None else {}
                ax.fill(*l, **fill_kwargs)
        ax.scatter(self.massifs_coordinates.x_coordinates, self.massifs_coordinates.y_coordinates)
        ax.axis('off')

        if show:
            plt.show()

    """ Some properties """

    @property
    def relative_path(self) -> str:
        return r'local/spatio_temporal_datasets'

    @property
    def full_path(self) -> str:
        return get_full_path(relative_path=self.relative_path)

    @property
    def safran_full_path(self) -> str:
        assert self.model_name in ['Safran', 'Crocus']
        return op.join(self.full_path, 'safran-crocus_{}'.format(self.altitude), self.model_name)

    @property
    def map_full_path(self) -> str:
        return op.join(self.full_path, 'map')

    @property
    def result_full_path(self) -> str:
        return op.join(self.full_path, 'results')
