import os
import os.path as op
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

from extreme_estimator.gev.gevmle_fit import GevMleFit
from safran_study.massif import safran_massif_names_from_datasets
from safran_study.snowfall_annual_maxima import SafranSnowfall
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from utils import get_full_path, cached_property


class Safran(object):

    def __init__(self, safran_altitude=1800, nb_days_of_snowfall=1):
        assert safran_altitude in [1800, 2400]
        self.safran_altitude = safran_altitude
        self.nb_days_of_snowfall = nb_days_of_snowfall

    def write_to_file(self, df):
        if not op.exists(self.result_full_path):
            os.makedirs(self.result_full_path, exist_ok=True)
        df.to_csv(op.join(self.result_full_path, 'merged_array_{}_altitude.csv'.format(self.safran_altitude)))

    """ Visualization methods """

    def visualize(self):
        df_massif = pd.read_csv(op.join(self.map_full_path, 'massifsalpes.csv'))
        coord_tuples = [(row_massif['idx'], row_massif[AbstractCoordinates.COORDINATE_X],
                         row_massif[AbstractCoordinates.COORDINATE_Y])
                        for _, row_massif in df_massif.iterrows()]
        for massif_idx in set([tuple[0] for tuple in coord_tuples]):
            l = [coords for idx, *coords in coord_tuples if idx == massif_idx]
            l = list(zip(*l))
            plt.plot(*l, color='black')
            plt.fill(*l)
        self.massifs_coordinates.visualization_2D()

    """ Statistics methods """

    @property
    def df_gev_mle_each_massif(self):
        # Fit a gev n each massif
        massif_to_gev_mle = {massif_name: GevMleFit(self.df_annual_maxima[massif_name]).gev_params.to_serie()
                             for massif_name in self.safran_massif_names}
        return pd.DataFrame(massif_to_gev_mle, columns=self.safran_massif_names)

    """ Annual maxima of snowfall """

    @property
    def df_annual_maxima(self):
        return pd.DataFrame(self.year_to_annual_maxima, index=self.safran_massif_names).T

    """ Load some attributes only once """

    @cached_property
    def year_to_annual_maxima(self):
        year_to_safran_snowfall = {year: SafranSnowfall(dataset) for year, dataset in
                                   self.year_to_dataset_ordered_dict.items()}
        year_to_annual_maxima = OrderedDict()
        for year in self.year_to_dataset_ordered_dict.keys():
            year_to_annual_maxima[year] = year_to_safran_snowfall[year].annual_maxima_of_snowfall(
                self.nb_days_of_snowfall)
        return year_to_annual_maxima

    @property
    def safran_massif_names(self):
        # Load the names of the massif as defined by SAFRAN
        return safran_massif_names_from_datasets(self.year_to_dataset_ordered_dict.values())

    @cached_property
    def year_to_dataset_ordered_dict(self) -> OrderedDict:
        # Map each year to the correspond netCDF4 Dataset
        year_to_dataset = OrderedDict()
        nc_files = [(int(f.split('_')[1][:4]), f) for f in os.listdir(self.safran_full_path) if f.endswith('.nc')]
        for year, nc_file in sorted(nc_files, key=lambda t: t[0]):
            year_to_dataset[year] = Dataset(op.join(self.safran_full_path, nc_file))
        return year_to_dataset

    @cached_property
    def massifs_coordinates(self) -> AbstractSpatialCoordinates:
        # Coordinate object that represents the massif coordinates in Lambert extended
        df_centroid = self.load_df_centroid()
        for coord_column in [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y]:
            df_centroid.loc[:, coord_column] = df_centroid[coord_column].str.replace(',', '.').astype(float)
        # Assert that the massif names are the same between SAFRAN and the coordinate file
        assert not set(self.safran_massif_names).symmetric_difference(set(df_centroid['NOM']))
        # Build coordinate object from df_centroid
        return AbstractSpatialCoordinates.from_df(df_centroid)

    def load_df_centroid(self):
        return pd.read_csv(op.join(self.map_full_path, 'coordonnees_massifs_alpes.csv'))

    """ Some properties """

    @property
    def relative_path(self) -> str:
        return r'local/spatio_temporal_datasets'

    @property
    def full_path(self) -> str:
        return get_full_path(relative_path=self.relative_path)

    @property
    def safran_full_path(self) -> str:
        return op.join(self.full_path, 'safran-crocus_{}'.format(self.safran_altitude), 'Safran')

    @property
    def map_full_path(self) -> str:
        return op.join(self.full_path, 'map')

    @property
    def result_full_path(self) -> str:
        return op.join(self.full_path, 'results')