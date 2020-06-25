import pandas as pd
import numpy as np
from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.slicer.utils import get_slicer_class_from_s_splits
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
import matplotlib.pyplot as plt


class AltitudesStudies(object):

    def __init__(self, study_class, altitudes,
                 spatial_transformation_class=None, temporal_transformation_class=None,
                 **kwargs_study):
        self.study_class = study_class
        self.spatial_transformation_class = spatial_transformation_class
        self.temporal_transformation_class = temporal_transformation_class
        self.altitudes = altitudes
        self.altitude_to_study = OrderedDict()  # type: OrderedDict[int, AbstractStudy]
        for altitude in self.altitudes:
            study = study_class(altitude=altitude, **kwargs_study)
            self.altitude_to_study[altitude] = study

    @cached_property
    def study(self) -> AbstractStudy:
        return list(self.altitude_to_study.values())[0]

    # Dataset Loader

    def spatio_temporal_dataset(self, massif_name, s_split_spatial: pd.Series = None,
                                s_split_temporal: pd.Series = None):
        coordinate_values_to_maxima = {}
        massif_altitudes = []
        for altitude in self.altitudes:
            study = self.altitude_to_study[altitude]
            if massif_name in study.study_massif_names:
                massif_altitudes.append(altitude)
                for year, maxima in zip(study.ordered_years, study.massif_name_to_annual_maxima[massif_name]):
                    coordinate_values_to_maxima[(altitude, year)] = [maxima]
        coordinates = self.spatio_temporal_coordinates(s_split_spatial, s_split_temporal, massif_altitudes)
        observations = AnnualMaxima.from_coordinates(coordinates, coordinate_values_to_maxima)
        return AbstractDataset(observations=observations, coordinates=coordinates)

    # Coordinates Loader

    def spatio_temporal_coordinates(self, s_split_spatial: pd.Series = None, s_split_temporal: pd.Series = None,
                                    massif_altitudes=None):
        if massif_altitudes is None or set(massif_altitudes) == set(self.altitudes):
            spatial_coordinates = self.spatial_coordinates
        else:
            spatial_coordinates = self.spatial_coordinates_for_altitudes(massif_altitudes)
        slicer_class = get_slicer_class_from_s_splits(s_split_spatial, s_split_temporal)
        return AbstractSpatioTemporalCoordinates(slicer_class=slicer_class,
                                                 s_split_spatial=s_split_spatial,
                                                 s_split_temporal=s_split_temporal,
                                                 transformation_class=self.spatial_transformation_class,
                                                 spatial_coordinates=spatial_coordinates,
                                                 temporal_coordinates=self.temporal_coordinates)

    @cached_property
    def temporal_coordinates(self):
        return ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=self.study.nb_years,
                                                                     start=self.study.year_min,
                                                                     transformation_class=self.spatial_transformation_class)

    @cached_property
    def spatial_coordinates(self):
        return self.spatial_coordinates_for_altitudes(self.altitudes)

    def spatial_coordinates_for_altitudes(self, altitudes):
        return AbstractSpatialCoordinates.from_list_x_coordinates(x_coordinates=altitudes,
                                                                  transformation_class=self.temporal_transformation_class)

    @cached_property
    def _df_coordinates(self):
        return AbstractSpatioTemporalCoordinates.get_df_from_spatial_and_temporal_coordinates(self.spatial_coordinates,
                                                                                              self.temporal_coordinates)

    def random_s_split_spatial(self, train_split_ratio):
        return AbstractCoordinates.spatial_s_split_from_df(self._df_coordinates, train_split_ratio)

    def random_s_split_temporal(self, train_split_ratio):
        return AbstractCoordinates.temporal_s_split_from_df(self._df_coordinates, train_split_ratio)

    # Some visualization

    def show_or_save_to_file(self, plot_name, show=False):
        study_visualizer = StudyVisualizer(study=self.study, show=show, save_to_file=not show)
        study_visualizer.plot_name = plot_name
        study_visualizer.show_or_save_to_file(add_classic_title=False, dpi=500)

    def run_for_each_massif(self, function, massif_names, **kwargs):
        massif_names = massif_names if massif_names is not None else self.study.all_massif_names()
        assert isinstance(massif_names, list)
        for i, massif_name in enumerate(massif_names):
            function(massif_name, massif_id=i, **kwargs)

    def plot_maxima_time_series(self, massif_names=None, show=False):
        self.run_for_each_massif(self._plot_maxima_time_series, massif_names, show=show)

    def _plot_maxima_time_series(self, massif_name, massif_id, show=False):
        ax = plt.gca()
        x = self.study.ordered_years
        for altitude, study in list(self.altitude_to_study.items())[::-1]:
            if massif_name in study.massif_name_to_annual_maxima:
                y = study.massif_name_to_annual_maxima[massif_name]
                label = '{} m'.format(altitude)
                ax.plot(x, y, linewidth=2, label=label)
        ax.xaxis.set_ticks(x[1::10])
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.legend()
        plot_name = 'Annual maxima of {} in {}'.format(SCM_STUDY_CLASS_TO_ABBREVIATION[self.study_class], massif_name)
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('years', fontsize=15)
        self.show_or_save_to_file(plot_name=plot_name, show=show)
        ax.clear()

    def plot_mean_maxima_against_altitude(self, massif_names=None, show=False, std=False, change=False):
        ax = plt.gca()
        self.run_for_each_massif(self._plot_mean_maxima_against_altitude, massif_names, ax=ax, std=std, change=change)
        ax.legend(prop={'size': 7}, ncol=3)
        moment = ''
        if change is None:
            moment += ' Relative'
        if change is True or change is None:
            moment += ' change (between two block of 30 years) for'
        moment += 'mean' if not std else 'std'
        plot_name = '{} of annual maxima of {}'.format(moment, SCM_STUDY_CLASS_TO_ABBREVIATION[self.study_class])
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('altitudes', fontsize=15)
        lim_down, lim_up = ax.get_ylim()
        lim_up += (lim_up - lim_down) / 3
        ax.set_ylim([lim_down, lim_up])
        ax.tick_params(axis='both', which='major', labelsize=13)
        self.show_or_save_to_file(plot_name=plot_name, show=show)
        ax.clear()

    def _plot_mean_maxima_against_altitude(self, massif_name, massif_id, ax=None, std=False, change=False):
        assert ax is not None
        altitudes = []
        mean_moment = []
        for altitude, study in self.altitude_to_study.items():
            if massif_name in study.massif_name_to_annual_maxima:
                annual_maxima = study.massif_name_to_annual_maxima[massif_name]
                function = np.std if std else np.mean
                if change in [True, None]:
                    after = function(annual_maxima[31:])
                    before = function(annual_maxima[:31])
                    moment = after - before
                    if change is None:
                        moment /= before
                        moment *= 100
                else:
                    moment = function(annual_maxima)
                mean_moment.append(moment)
                altitudes.append(altitude)
        plot_against_altitude(altitudes, ax, massif_id, massif_name, mean_moment)
