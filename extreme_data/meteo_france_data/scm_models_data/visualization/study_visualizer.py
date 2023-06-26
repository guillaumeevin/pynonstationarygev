import os
import os.path as op

import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import load_plot
from extreme_fit.function.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_fit.function.param_function.param_function import AbstractParamFunction
from root_utils import SHORT_VERSION_TIME
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset

BLOCK_MAXIMA_DISPLAY_NAME = 'block maxima '


class VisualizationParameters(object):

    def __init__(self, save_to_file=False, only_one_graph=False, only_first_row=False, show=True):
        self.only_first_row = only_first_row
        self.only_one_graph = only_one_graph
        self.save_to_file = save_to_file

        # PLOT ARGUMENTS
        self.show = False if self.save_to_file else show
        if self.only_one_graph:
            self.figsize = (6.0, 4.0)
        elif self.only_first_row:
            self.figsize = (8.0, 6.0)
        else:
            self.figsize = (16.0, 10.0)
        self.subplot_space = 0.5
        self.coef_zoom_map = 1


class StudyVisualizer(VisualizationParameters):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False):
        super().__init__(save_to_file, only_one_graph, only_first_row, show)
        self.nb_cores = 7
        self.massif_id_to_smooth_maxima = {}
        self.temporal_non_stationarity = temporal_non_stationarity
        self.only_first_row = only_first_row
        self.only_one_graph = only_one_graph
        self.save_to_file = save_to_file
        self.study = study
        self.plot_name = None

        self.multiprocessing = multiprocessing
        self.verbose = verbose
        self.complete_non_stationary_trend_analysis = complete_non_stationary_trend_analysis

        # Load some attributes
        self._dataset = None
        self._coordinates = None
        self._observations = None

        # KDE PLOT ARGUMENTS
        self.vertical_kde_plot = vertical_kde_plot
        self.year_for_kde_plot = year_for_kde_plot
        self.plot_block_maxima_quantiles = plot_block_maxima_quantiles

        self.window_size_for_smoothing = 1  # other value could be
        self.number_of_top_values = 10  # 1 if we just want the maxima

        # Modify some class attributes
        # Remove some assert
        AbstractParamFunction.OUT_OF_BOUNDS_ASSERT = False
        # INCREASE THE TEMPORAL STEPS FOR VISUALIZATION
        AbstractMarginFunction.VISUALIZATION_TEMPORAL_STEPS = 5

        # Change point parameters
        self.trend_test_class_for_change_point_test = None
        self.starting_years_for_change_point_test = None
        self.nb_massif_for_change_point_test = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = AbstractDataset(self.observations, self.coordinates)
        return self._dataset

    @property
    def spatial_coordinates(self):
        return AbstractSpatialCoordinates.from_df(df=self.study.df_massifs_longitude_and_latitude)

    @property
    def temporal_coordinates(self):
        start, stop = self.study.start_year_and_stop_year
        nb_steps = stop - start + 1
        temporal_coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=nb_steps,
                                                                                     start=start)
        return temporal_coordinates

    @property
    def spatio_temporal_coordinates(self):
        return AbstractSpatioTemporalCoordinates.from_spatial_coordinates_and_temporal_coordinates(
            spatial_coordinates=self.spatial_coordinates, temporal_coordinates=self.temporal_coordinates)

    @property
    def coordinates(self):
        if self._coordinates is None:
            if self.temporal_non_stationarity:
                # Build spatio temporal coordinates from a spatial coordinates and a temporal coordinates
                coordinates = self.spatio_temporal_coordinates
            else:
                # By default otherwise, we only keep the spatial coordinates
                coordinates = self.spatial_coordinates
            self._coordinates = coordinates
        return self._coordinates

    @property
    def observations(self):
        if self._observations is None:
            self._observations = self.study.observations_annual_maxima
            if self.temporal_non_stationarity:
                self._observations.convert_to_spatio_temporal_index(self.coordinates)
                if self.verbose:
                    self._observations.print_summary()
        return self._observations

    def show_or_save_to_file(self, add_classic_title=False, no_title=True, tight_layout=False, tight_pad=None,
                             dpi=None, folder_for_variable=True, plot_name=None):
        if plot_name is not None:
            self.plot_name = plot_name

        if isinstance(self.study, AbstractAdamontStudy):
            prefix = gcm_rcm_couple_to_str(self.study.gcm_rcm_couple)
            prefix = prefix.replace('/', '-')
            self.plot_name = prefix + ' ' + self.plot_name

        assert self.plot_name is not None
        if add_classic_title:
            title = self.study.title
            title += '\n' + self.plot_name
        else:
            title = self.plot_name
        if self.only_one_graph:
            plt.suptitle(self.plot_name,  y=1.0)
        elif not no_title:
            plt.suptitle(title,  y=1.0)
        if self.show:
            plt.show()
        if self.save_to_file:
            main_title, specific_title = '_'.join(self.study.title.split()).split('/')
            main_title += self.study.season_name
            # Shorter main title
            main_title = '_'.join(main_title.split('_')[:2])
            if folder_for_variable:
                filename = "{}/{}/".format(SHORT_VERSION_TIME, main_title)
            else:
                filename = "{}/".format(SHORT_VERSION_TIME)
            if not self.only_one_graph:
                filename += "{}".format('_'.join(self.plot_name.split())) + '_'
            filename += specific_title
            # Save a first time in transparent
            self.savefig_in_results(filename, transparent=True)
            self.savefig_in_results(filename, transparent=False, tight_pad=tight_pad)

    @classmethod
    def savefig_in_results(cls, filename, transparent=True, tight_pad=None):
        img_format = 'svg' if transparent else 'png'
        filepath = op.join(AbstractStudy.result_full_path, filename + '.' + img_format)
        if transparent:
            dir_list = filepath.split('/')
            dir_list[-1:] = ['svg', dir_list[-1]]
            filepath = '/'.join(dir_list)
        dirname = op.dirname(filepath)
        if not op.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        if transparent:
            plt.savefig(filepath, bbox_inches=0, transparent=True)
        else:
            if tight_pad is not None:
                plt.tight_layout(**tight_pad)
            else:
                plt.tight_layout()
            plt.savefig(filepath, bbox_inches=0, transparent=False)

            
        # if dpi is not None:
        #     plt.savefig(filepath, dpi=dpi)
        # else:
        #     plt.savefig(filepath)




    """ Statistics methods """


    # PLot functions that should be common

    def plot_map(self, cmap, graduation, label, massif_name_to_value, plot_name, add_x_label=True,
                 negative_and_positive_values=True, massif_name_to_text=None, altitude=None, add_colorbar=True,
                 max_abs_change=None, xlabel=None, fontsize_label=10, massif_names_with_white_dot=None,
                 half_cmap_for_positive=True):
        if altitude is None:
            altitude = self.study.altitude
        if len(massif_name_to_value) > 0:
            load_plot(cmap, graduation, label, massif_name_to_value, altitude,
                      add_x_label=add_x_label, negative_and_positive_values=negative_and_positive_values,
                      massif_name_to_text=massif_name_to_text,
                      add_colorbar=add_colorbar, max_abs_change=max_abs_change, xlabel=xlabel,
                      fontsize_label=fontsize_label, massif_names_with_white_dot=massif_names_with_white_dot,
                      half_cmap_for_positive=half_cmap_for_positive)
            self.plot_name = plot_name
            # self.show_or_save_to_file(add_classic_title=False, tight_layout=True, no_title=True, dpi=500)
            self.show_or_save_to_file(add_classic_title=False, no_title=True, dpi=500, tight_layout=True)
            plt.close()


