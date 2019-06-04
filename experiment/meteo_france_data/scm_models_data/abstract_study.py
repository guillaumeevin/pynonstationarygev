import io
import os
import os.path as op
from collections import OrderedDict
from contextlib import redirect_stdout
from multiprocessing.pool import Pool
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from matplotlib import cm
from matplotlib.colors import Normalize
from netCDF4 import Dataset

from experiment.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from experiment.meteo_france_data.scm_models_data.scm_constants import ALTITUDES, ZS_INT_23, ZS_INT_MASK, LONGITUDES, \
    LATITUDES
from experiment.meteo_france_data.scm_models_data.visualization.utils import get_km_formatter
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.margin_fits.plot.create_shifted_cmap import get_color_rbga_shifted, create_colorbase_axis
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
from utils import get_full_path, cached_property, NB_CORES

f = io.StringIO()
with redirect_stdout(f):
    from simpledbf import Dbf5


class AbstractStudy(object):
    """
    A Study is defined by:
        - a variable class that correspond to the meteorogical quantity of interest
        - an altitude of interest
        - a start and a end year

    Les fichiers netcdf de SAFRAN et CROCUS sont autodocumentés (on peut les comprendre avec ncdump -h notamment).
    """
    REANALYSIS_FOLDER = 'alp_flat/reanalysis'

    def __init__(self, variable_class: type, altitude: int = 1800, year_min=1000, year_max=3000,
                 multiprocessing=True):
        assert altitude in ALTITUDES, altitude
        self.altitude = altitude
        self.model_name = None
        self.variable_class = variable_class
        self.year_min = year_min
        self.year_max = year_max
        self.multiprocessing = multiprocessing

    """ Annual maxima """

    @property
    def observations_annual_maxima(self) -> AnnualMaxima:
        return AnnualMaxima(df_maxima_gev=pd.DataFrame(self.year_to_annual_maxima, index=self.study_massif_names))

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_maxima = OrderedDict()
        for year, time_serie in self._year_to_max_daily_time_serie.items():
            year_to_annual_maxima[year] = time_serie.max(axis=0)
        return year_to_annual_maxima

    """ Annual total """

    @property
    def df_annual_total(self) -> pd.DataFrame:
        return pd.DataFrame(self.year_to_annual_total, index=self.study_massif_names).transpose()

    def annual_aggregation_function(self, *args, **kwargs):
        raise NotImplementedError()

    @cached_property
    def year_to_annual_total(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_mean = OrderedDict()
        for year, time_serie in self._year_to_daily_time_serie_array.items():
            year_to_annual_mean[year] = self.apply_annual_aggregation(time_serie)
        return year_to_annual_mean

    def apply_annual_aggregation(self, time_serie):
        return self.annual_aggregation_function(time_serie, axis=0)

    """ Load daily observations """

    @cached_property
    def year_to_daily_time_serie_array(self) -> OrderedDict:
        return self._year_to_daily_time_serie_array

    @property
    def _year_to_max_daily_time_serie(self) -> OrderedDict:
        return self._year_to_daily_time_serie_array

    @property
    def _year_to_daily_time_serie_array(self) -> OrderedDict:
        # Map each year to a matrix of size 365-nb_days_consecutive+1 x nb_massifs
        year_to_daily_time_serie_array = OrderedDict()
        for year in self.ordered_years:
            # Check daily data
            daily_time_serie = self.year_to_variable_object[year].daily_time_serie_array
            assert daily_time_serie.shape[0] in [365, 366]
            assert daily_time_serie.shape[1] == len(ZS_INT_MASK)
            # Filter only the data corresponding to the altitude of interest
            daily_time_serie = daily_time_serie[:, self.altitude_mask]
            year_to_daily_time_serie_array[year] = daily_time_serie
        return year_to_daily_time_serie_array



    """ Load Variables and Datasets """

    @cached_property
    def year_to_variable_object(self) -> OrderedDict:
        # Map each year to the variable array
        path_files, ordered_years = self.ordered_years_and_path_files
        if self.multiprocessing:
            with Pool(NB_CORES) as p:
                variables = p.map(self.load_variable_object, path_files)
        else:
            variables = [self.load_variable_object(path_file) for path_file in path_files]
        return OrderedDict(zip(ordered_years, variables))

    def instantiate_variable_object(self, variable_array) -> AbstractVariable:
        return self.variable_class(variable_array)

    def load_variable_array(self, dataset):
        return np.array(dataset.variables[self.load_keyword()])

    def load_variable_object(self, path_file):
        dataset = Dataset(path_file)
        variable_array = self.load_variable_array(dataset)
        return self.instantiate_variable_object(variable_array)

    def load_keyword(self):
        return self.variable_class.keyword()

    @property
    def year_to_dataset_ordered_dict(self) -> OrderedDict:
        print('This code is quite long... '
              'You should consider year_to_variable which is way faster when multiprocessing=True')
        # Map each year to the correspond netCDF4 Dataset
        path_files, ordered_years = self.ordered_years_and_path_files
        datasets = [Dataset(path_file) for path_file in path_files]
        return OrderedDict(zip(ordered_years, datasets))

    @cached_property
    def ordered_years_and_path_files(self):
        nc_files = [(int(f.split('_')[-2][:4]), f) for f in os.listdir(self.study_full_path) if f.endswith('.nc')]
        ordered_years, path_files = zip(*[(year, op.join(self.study_full_path, nc_file))
                                          for year, nc_file in sorted(nc_files, key=lambda t: t[0])
                                          if self.year_min <= year < self.year_max])
        return path_files, ordered_years

    """ Temporal properties """

    @property
    def ordered_years(self):
        return self.ordered_years_and_path_files[1]

    @property
    def start_year_and_stop_year(self) -> Tuple[int, int]:
        ordered_years = self.ordered_years
        return ordered_years[0], ordered_years[-1]

    """ Spatial properties """

    @property
    def study_massif_names(self) -> List[str]:
        return self.altitude_to_massif_names[self.altitude]

    @property
    def df_massifs_longitude_and_latitude(self) -> pd.DataFrame:
        # DataFrame object that represents the massif coordinates in degrees extracted from the SCM data
        # Another way of getting the latitudes and longitudes could have been the following:
        # any_ordered_dict = list(self.year_to_dataset_ordered_dict.values())[0]
        # longitude = np.array(any_ordered_dict.variables['longitude'])
        # latitude = np.array(any_ordered_dict.variables['latitude'])
        longitude = np.array(LONGITUDES)
        latitude = np.array(LATITUDES)
        columns = [AbstractSpatialCoordinates.COORDINATE_X, AbstractSpatialCoordinates.COORDINATE_Y]
        data = dict(zip(columns, [longitude[self.altitude_mask], latitude[self.altitude_mask]]))
        return pd.DataFrame(data=data, index=self.study_massif_names, columns=columns)

    @property
    def missing_massif_name(self):
        return set(self.all_massif_names) - set(self.altitude_to_massif_names[self.altitude])

    @cached_property
    def altitude_mask(self):
        altitude_mask = ZS_INT_MASK == self.altitude
        assert np.sum(altitude_mask) == len(self.altitude_to_massif_names[self.altitude])
        return altitude_mask

    """ Path properties """

    @property
    def title(self):
        return "{}/at altitude {}m ({} mountain chains)".format(self.variable_name, self.altitude,
                                                                len(self.study_massif_names))

    @property
    def variable_name(self):
        return self.variable_class.NAME + ' (in {})'.format(self.variable_unit)

    @property
    def variable_unit(self):
        return self.variable_class.UNIT

    """ Visualization methods """

    @cached_property
    def massifs_coordinates_for_display(self) -> AbstractSpatialCoordinates:
        # Coordinate object that represents the massif coordinates in Lambert extended
        # extracted for a csv file, and used only for display purposes
        df = self.load_df_centroid()
        # Filter, keep massifs present at the altitude of interest
        df = df.loc[self.study_massif_names]
        # Build coordinate object from df_centroid
        return AbstractSpatialCoordinates.from_df(df)

    def visualize_study(self, ax=None, massif_name_to_value=None, show=True, fill=True, replace_blue_by_white=True,
                        label=None, add_text=False, cmap=None, vmax=100, vmin=0):
        if massif_name_to_value is None:
            massif_name_to_fill_kwargs = None
        else:
            massif_names, values = list(zip(*massif_name_to_value.items()))
            if cmap is None:
                colors = get_color_rbga_shifted(ax, replace_blue_by_white, values, label=label)
            else:
                norm = Normalize(vmin, vmax)
                create_colorbase_axis(ax, label, cmap, norm)
                m = cm.ScalarMappable(norm=norm, cmap=cmap)
                colors = [m.to_rgba(value) if not np.isnan(value) else 'w' for value in values]
            massif_name_to_fill_kwargs = {massif_name: {'color': color} for massif_name, color in
                                          zip(massif_names, colors)}

        if ax is None:
            ax = plt.gca()

        for coordinate_id, coords_list in self.idx_to_coords_list.items():
            # Retrieve the list of coords (x,y) that define the contour of the massif of id coordinate_id

            # if j == 0:
            #     mask_outside_polygon(poly_verts=l, ax=ax)
            # Plot the contour of the massif
            coords_list = list(zip(*coords_list))
            ax.plot(*coords_list, color='black')
            # Potentially, fill the inside of the polygon with some color
            if fill and coordinate_id in self.coordinate_id_to_massif_name:
                massif_name = self.coordinate_id_to_massif_name[coordinate_id]
                if massif_name_to_fill_kwargs is not None and massif_name in massif_name_to_fill_kwargs:
                    fill_kwargs = massif_name_to_fill_kwargs[massif_name]
                    ax.fill(*coords_list, **fill_kwargs)
                # else:
                #     fill_kwargs = {}

                # x , y = list(self.massifs_coordinates.df_all_coordinates.loc[massif_name])
                # x , y= coords_list[0][0], coords_list[0][1]
                # print(x, y)
                # print(massif_name)
                # ax.scatter(x, y)
                # ax.text(x, y, massif_name)
        # Display the center of the massif
        ax.scatter(self.massifs_coordinates_for_display.x_coordinates,
                   self.massifs_coordinates_for_display.y_coordinates, s=1)
        # Improve some explanation on the X axis and on the Y axis
        ax.set_xlabel('Longitude (km)')
        ax.xaxis.set_major_formatter(get_km_formatter())
        ax.set_ylabel('Latitude (km)')
        ax.yaxis.set_major_formatter(get_km_formatter())
        # Display the name or value of the massif
        if add_text:
            for _, row in self.massifs_coordinates_for_display.df_all_coordinates.iterrows():
                x, y = list(row)
                massif_name = row.name
                value = massif_name_to_value[massif_name]
                ax.text(x, y, str(round(value, 1)))

        if show:
            plt.show()

    """ 
    CLASS ATTRIBUTES COMMON TO ALL OBJECTS 
    (written as object attributes/methods for simplicity)
    """

    """ Path properties """

    @property
    def relative_path(self) -> str:
        return r'local/spatio_temporal_datasets'

    @property
    def full_path(self) -> str:
        return get_full_path(relative_path=self.relative_path)

    @property
    def map_full_path(self) -> str:
        return op.join(self.full_path, 'map')

    @property
    def result_full_path(self) -> str:
        return op.join(self.full_path, 'results')

    @property
    def study_full_path(self) -> str:
        assert self.model_name in ['Safran', 'Crocus']
        study_folder = 'meteo' if self.model_name is 'Safran' else 'pro'
        return op.join(self.full_path, self.REANALYSIS_FOLDER, study_folder)

    """  Spatial properties """

    @property
    def original_safran_massif_id_to_massif_name(self) -> Dict[int, str]:
        return {massif_id: massif_name for massif_id, massif_name in enumerate(self.all_massif_names)}

    @cached_property
    def all_massif_names(self) -> List[str]:
        """
        Pour l'identification des massifs, le numéro de la variable massif_num correspond à celui de l'attribut num_opp
        """
        metadata_path = op.join(self.full_path, self.REANALYSIS_FOLDER, 'metadata')
        dbf = Dbf5(op.join(metadata_path, 'massifs_alpes.dbf'))
        df = dbf.to_dataframe().copy()  # type: pd.DataFrame
        dbf.f.close()
        df.sort_values(by='num_opp', inplace=True)
        all_massif_names = list(df['nom'])
        # Correct a massif name
        all_massif_names[all_massif_names.index('Beaufortin')] = 'Beaufortain'
        return all_massif_names

    def load_df_centroid(self) -> pd.DataFrame:
        # Load df_centroid containing all the massif names
        df_centroid = pd.read_csv(op.join(self.map_full_path, 'coordonnees_massifs_alpes.csv'))
        df_centroid.set_index('NOM', inplace=True)
        # Check that the names of massifs are the same
        symmetric_difference = set(df_centroid.index).symmetric_difference(self.all_massif_names)
        assert len(symmetric_difference) == 0, symmetric_difference
        # Sort the column in the order of the SAFRAN dataset
        df_centroid = df_centroid.reindex(self.all_massif_names, axis=0)
        for coord_column in [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y]:
            df_centroid.loc[:, coord_column] = df_centroid[coord_column].str.replace(',', '.').astype(float)
        return df_centroid

    @cached_property
    def massif_name_to_altitudes(self) -> Dict[str, List[int]]:
        s = ZS_INT_23 + [0]
        zs_list = []
        zs_all_list = []
        for a, b in zip(s[:-1], s[1:]):
            zs_list.append(a)
            if a > b:
                zs_all_list.append(zs_list)
                zs_list = []
        return dict(zip(self.all_massif_names, zs_all_list))

    @cached_property
    def altitude_to_massif_names(self) -> Dict[int, List[str]]:
        altitude_to_massif_names = {altitude: [] for altitude in ALTITUDES}
        for massif_name in self.massif_name_to_altitudes.keys():
            for altitude in self.massif_name_to_altitudes[massif_name]:
                altitude_to_massif_names[altitude].append(massif_name)
        return altitude_to_massif_names

    """ Visualization methods """

    @property
    def coordinate_id_to_massif_name(self) -> Dict[int, str]:
        df_centroid = self.load_df_centroid()
        return dict(zip(df_centroid['id'], df_centroid.index))

    @property
    def idx_to_coords_list(self):
        df_massif = pd.read_csv(op.join(self.map_full_path, 'massifsalpes.csv'))
        coord_tuples = [(row_massif['idx'], row_massif[AbstractCoordinates.COORDINATE_X],
                         row_massif[AbstractCoordinates.COORDINATE_Y])
                        for _, row_massif in df_massif.iterrows()]
        all_idxs = set([t[0] for t in coord_tuples])
        return {idx: [coords for idx_loop, *coords in coord_tuples if idx == idx_loop] for idx in all_idxs}

    @property
    def all_coords_list(self):
        all_values = []
        for e in self.idx_to_coords_list.values():
            all_values.extend(e)
        return list(zip(*all_values))

    @property
    def visualization_x_limits(self):
        return min(self.all_coords_list[0]), max(self.all_coords_list[0])

    @property
    def visualization_y_limits(self):
        return min(self.all_coords_list[1]), max(self.all_coords_list[1])

    @cached_property
    def mask_french_alps(self):
        resolution = AbstractMarginFunction.VISUALIZATION_RESOLUTION
        mask_french_alps = np.zeros([resolution, resolution])
        for polygon in self.idx_to_coords_list.values():
            xy_values = list(zip(*polygon))
            normalized_polygon = []
            for values, (minlim, max_lim) in zip(xy_values, [self.visualization_x_limits, self.visualization_y_limits]):
                values -= minlim
                values *= resolution / (max_lim - minlim)
                normalized_polygon.append(values)
            normalized_polygon = list(zip(*normalized_polygon))
            img = Image.new('L', (resolution, resolution), 0)
            ImageDraw.Draw(img).polygon(normalized_polygon, outline=1, fill=1)
            mask_massif = np.array(img)
            mask_french_alps += mask_massif
        return ~np.array(mask_french_alps, dtype=bool)
