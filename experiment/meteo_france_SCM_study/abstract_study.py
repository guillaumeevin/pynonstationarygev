import os
import os.path as op
from collections import OrderedDict
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable
from experiment.meteo_france_SCM_study.massif import safran_massif_names_from_datasets
from experiment.meteo_france_SCM_study.visualization.utils import get_km_formatter
from extreme_estimator.margin_fits.plot.create_shifted_cmap import get_color_rbga_shifted
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
from utils import get_full_path, cached_property


class AbstractStudy(object):
    ALTITUDES = [1800, 2400]

    def __init__(self, variable_class: type, altitude: int = 1800, year_min=1000, year_max=3000):
        assert altitude in self.ALTITUDES, altitude
        self.altitude = altitude
        self.model_name = None
        self.variable_class = variable_class
        self.year_min = year_min
        self.year_max = year_max

    def write_to_file(self, df: pd.DataFrame):
        if not op.exists(self.result_full_path):
            os.makedirs(self.result_full_path, exist_ok=True)
        df.to_csv(op.join(self.result_full_path, 'merged_array_{}_altitude.csv'.format(self.altitude)))

    """ Data """

    @property
    def df_all_daily_time_series_concatenated(self) -> pd.DataFrame:
        df_list = [pd.DataFrame(time_serie, columns=self.safran_massif_names) for time_serie in
                   self.year_to_daily_time_serie_array.values()]
        df_concatenated = pd.concat(df_list)
        return df_concatenated

    @property
    def observations_annual_maxima(self) -> AnnualMaxima:
        return AnnualMaxima(df_maxima_gev=pd.DataFrame(self.year_to_annual_maxima, index=self.safran_massif_names))

    @property
    def df_annual_total(self) -> pd.DataFrame:
        return pd.DataFrame(self.year_to_annual_total, index=self.safran_massif_names).transpose()

    def annual_aggregation_function(self, *args, **kwargs):
        raise NotImplementedError()

    """ Load some attributes only once """

    @cached_property
    def year_to_dataset_ordered_dict(self) -> OrderedDict:
        # Map each year to the correspond netCDF4 Dataset
        year_to_dataset = OrderedDict()
        nc_files = [(int(f.split('_')[-2][:4]), f) for f in os.listdir(self.safran_full_path) if f.endswith('.nc')]
        for year, nc_file in sorted(nc_files, key=lambda t: t[0]):
            if self.year_min <= year < self.year_max:
                year_to_dataset[year] = Dataset(op.join(self.safran_full_path, nc_file))
        return year_to_dataset

    @cached_property
    def year_to_daily_time_serie_array(self) -> OrderedDict:
        return self._year_to_daily_time_serie_array

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_maxima = OrderedDict()
        for year, time_serie in self._year_to_max_daily_time_serie.items():
            year_to_annual_maxima[year] = time_serie.max(axis=0)
        return year_to_annual_maxima

    @cached_property
    def year_to_annual_total(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_mean = OrderedDict()
        for year, time_serie in self._year_to_daily_time_serie_array.items():
            year_to_annual_mean[year] = self.apply_annual_aggregation(time_serie)
        return year_to_annual_mean

    def apply_annual_aggregation(self, time_serie):
        return self.annual_aggregation_function(time_serie, axis=0)

    def instantiate_variable_object(self, dataset) -> AbstractVariable:
        return self.variable_class(dataset, self.altitude)

    """ Private methods to be overwritten """

    @property
    def _year_to_daily_time_serie_array(self) -> OrderedDict:
        # Map each year to a matrix of size 365-nb_days_consecutive+1 x nb_massifs
        year_to_variable = {year: self.instantiate_variable_object(dataset) for year, dataset in
                            self.year_to_dataset_ordered_dict.items()}
        year_to_daily_time_serie_array = OrderedDict()
        for year in self.year_to_dataset_ordered_dict.keys():
            year_to_daily_time_serie_array[year] = year_to_variable[year].daily_time_serie_array
        return year_to_daily_time_serie_array

    @property
    def _year_to_max_daily_time_serie(self) -> OrderedDict:
        return self._year_to_daily_time_serie_array

    ##########

    @property
    def safran_massif_names(self) -> List[str]:
        return self.original_safran_massif_names

    @property
    def original_safran_massif_names(self) -> List[str]:
        # Load the names of the massif as defined by SAFRAN
        return safran_massif_names_from_datasets(list(self.year_to_dataset_ordered_dict.values()), self.altitude)

    @property
    def original_safran_massif_id_to_massif_name(self) -> Dict[int, str]:
        return {massif_id: massif_name for massif_id, massif_name in enumerate(self.original_safran_massif_names)}

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
        df_centroid.set_index('NOM', inplace=True)
        # Sort the column in the order of the SAFRAN dataset
        df_centroid = df_centroid.loc[self.original_safran_massif_names]
        return df_centroid

    @property
    def coordinate_id_to_massif_name(self) -> Dict[int, str]:
        df_centroid = self.load_df_centroid()
        return dict(zip(df_centroid['id'], df_centroid.index))

    """ Visualization methods """

    def visualize_study(self, ax=None, massif_name_to_value=None, show=True, fill=True, replace_blue_by_white=True,
                        label=None, add_text=False):
        if massif_name_to_value is None:
            massif_name_to_fill_kwargs = None
        else:
            massif_names, values = list(zip(*massif_name_to_value.items()))
            colors = get_color_rbga_shifted(ax, replace_blue_by_white, values, label=label)
            massif_name_to_fill_kwargs = {massif_name: {'color': color} for massif_name, color in
                                          zip(massif_names, colors)}

        if ax is None:
            ax = plt.gca()
        df_massif = pd.read_csv(op.join(self.map_full_path, 'massifsalpes.csv'))
        coord_tuples = [(row_massif['idx'], row_massif[AbstractCoordinates.COORDINATE_X],
                         row_massif[AbstractCoordinates.COORDINATE_Y])
                        for _, row_massif in df_massif.iterrows()]

        for _, coordinate_id in enumerate(set([t[0] for t in coord_tuples])):
            # Retrieve the list of coords (x,y) that define the contour of the massif of id coordinate_id
            coords_list = [coords for idx, *coords in coord_tuples if idx == coordinate_id]
            # if j == 0:
            #     mask_outside_polygon(poly_verts=l, ax=ax)
            # Plot the contour of the massif
            coords_list = list(zip(*coords_list))
            ax.plot(*coords_list, color='black')
            # Potentially, fill the inside of the polygon with some color
            if fill and coordinate_id in self.coordinate_id_to_massif_name:
                massif_name = self.coordinate_id_to_massif_name[coordinate_id]
                fill_kwargs = massif_name_to_fill_kwargs[massif_name] if massif_name_to_fill_kwargs is not None else {}
                ax.fill(*coords_list, **fill_kwargs)
                # x , y = list(self.massifs_coordinates.df_all_coordinates.loc[massif_name])
                # x , y= coords_list[0][0], coords_list[0][1]
                # print(x, y)
                # print(massif_name)
                # ax.scatter(x, y)
                # ax.text(x, y, massif_name)
        # Display the center of the massif
        ax.scatter(self.massifs_coordinates.x_coordinates, self.massifs_coordinates.y_coordinates, s=1)
        # Improve some explanation on the X axis and on the Y axis
        ax.set_xlabel('Longitude (km)')
        ax.xaxis.set_major_formatter(get_km_formatter())
        ax.set_ylabel('Latitude (km)')
        ax.yaxis.set_major_formatter(get_km_formatter())
        # Display the name or value of the massif
        if add_text:
            for _, row in self.massifs_coordinates.df_all_coordinates.iterrows():
                x, y = list(row)
                massif_name = row.name
                value = massif_name_to_value[massif_name]
                ax.text(x, y, str(round(value, 1)))

        if show:
            plt.show()

    """ Some properties """

    @property
    def title(self):
        return "{} at altitude {}m".format(self.variable_name, self.altitude)

    @property
    def variable_name(self):
        return self.variable_class.NAME + ' (in {})'.format(self.variable_unit)

    @property
    def variable_unit(self):
        return self.variable_class.UNIT

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
