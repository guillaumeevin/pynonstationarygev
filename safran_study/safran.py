import os
import os.path as op
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid
from netCDF4 import Dataset

from extreme_estimator.gev.gevmle_fit import GevMleFit
from extreme_estimator.gev_params import GevParams
from safran_study.massif import safran_massif_names_from_datasets
from safran_study.shifted_color_map import shiftedColorMap
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

    def visualize(self, ax=None, massif_name_to_fill_kwargs=None, show=True):
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
            massif_name = self.coordinate_id_to_massif_name[coordinate_id]
            fill_kwargs = massif_name_to_fill_kwargs[massif_name] if massif_name_to_fill_kwargs is not None else {}
            ax.fill(*l, **fill_kwargs)
        cax = ax.scatter(self.massifs_coordinates.x_coordinates, self.massifs_coordinates.y_coordinates)

        if show:
            plt.show()
        return cax

    def visualize_gev_fit_with_cmap(self, show=True, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, len(GevParams.GEV_PARAM_NAMES))
            fig.subplots_adjust(hspace=1.0, wspace=1.0)

            # fig = plt.figure(figsize=(6, 6))
            # axes = AxesGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.5,
            #                 label_mode="1", share_all=True,
            #                 cbar_location="right", cbar_mode="each",
            #                 cbar_size="7%", cbar_pad="2%")

        for i, gev_param_name in enumerate(GevParams.GEV_PARAM_NAMES[-1:]):
            massif_name_to_value = self.df_gev_mle_each_massif.loc[gev_param_name, :].to_dict()
            # Compute the middle point of the values for the color map
            values = list(massif_name_to_value.values())
            vmin, vmax = min(values), max(values)
            midpoint = 1 - vmax / (vmax + abs(vmin))
            scaling_factor = max(vmax, -vmin)
            # print(gev_param_name, midpoint, vmin, vmax, scaling_factor)
            # Load the shifted cmap to center on a middle point
            cmap = [plt.cm.coolwarm, plt.cm.bwr, plt.cm.seismic][0]
            shifted_cmap = shiftedColorMap(plt.cm.coolwarm, midpoint=0.0, name='shifted')
            massif_name_to_fill_kwargs = {massif_name: {'color': shifted_cmap(value / scaling_factor)} for massif_name, value in
                                          massif_name_to_value.items()}
            ax = axes[i]
            cax = self.visualize(ax=ax, massif_name_to_fill_kwargs=massif_name_to_fill_kwargs, show=False)

            # cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
            # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
            # cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
            # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
            title_str = gev_param_name
            ax.set_title(title_str)

        if show:
            plt.show()

    def visualize_cmap(self, massif_name_to_value):
        orig_cmap = plt.cm.coolwarm
        # shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.75, name='shifted')

        massif_name_to_fill_kwargs = {massif_name: {'color': orig_cmap(value)} for massif_name, value in massif_name_to_value.items()}

        self.visualize(massif_name_to_fill_kwargs=massif_name_to_fill_kwargs)

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

    @property
    def safran_massif_id_to_massif_name(self):
        return dict(enumerate(self.safran_massif_names))

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
        # Build coordinate object from df_centroid
        return AbstractSpatialCoordinates.from_df(df_centroid)

    def load_df_centroid(self) -> pd.DataFrame:
        df_centroid = pd.read_csv(op.join(self.map_full_path, 'coordonnees_massifs_alpes.csv'))
        # Assert that the massif names are the same between SAFRAN and the coordinate file
        assert not set(self.safran_massif_names).symmetric_difference(set(df_centroid['NOM']))
        return df_centroid

    @property
    def coordinate_id_to_massif_name(self):
        df_centroid = self.load_df_centroid()
        print(df_centroid.columns)
        return dict(zip(df_centroid['id'], df_centroid['NOM']))

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