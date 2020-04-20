import pandas as pd
from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
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
        coordinates = self.spatio_temporal_coordinates(s_split_spatial, s_split_temporal)
        coordinate_values_to_maxima = {}
        for altitude in self.altitudes:
            study = self.altitude_to_study[altitude]
            for year, maxima in zip(study.ordered_years, study.massif_name_to_annual_maxima[massif_name]):
                coordinate_values_to_maxima[(altitude, year)] = [maxima]
        observations = AnnualMaxima.from_coordinates(coordinates, coordinate_values_to_maxima)
        return AbstractDataset(observations=observations, coordinates=coordinates)

    # Coordinates Loader

    def spatio_temporal_coordinates(self, s_split_spatial: pd.Series = None, s_split_temporal: pd.Series = None):
        slicer_class = AbstractCoordinates.slicer_class_from_s_splits(s_split_spatial=s_split_spatial,
                                                                      s_split_temporal=s_split_temporal)
        return AbstractSpatioTemporalCoordinates(slicer_class=slicer_class,
                                                 s_split_spatial=s_split_spatial,
                                                 s_split_temporal=s_split_temporal,
                                                 transformation_class=self.spatial_transformation_class,
                                                 spatial_coordinates=self.spatial_coordinates,
                                                 temporal_coordinates=self.temporal_coordinates)

    @cached_property
    def temporal_coordinates(self):
        return ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=self.study.nb_years,
                                                                     start=self.study.year_min,
                                                                     transformation_class=self.spatial_transformation_class)

    @cached_property
    def spatial_coordinates(self):
        return AbstractSpatialCoordinates.from_list_x_coordinates(x_coordinates=self.altitudes,
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
        study_visualizer.show_or_save_to_file(add_classic_title=False)

    def plot_maxima_time_series(self, massif_names=None, show=False):
        massif_names = massif_names if massif_names is not None else self.study.all_massif_names()
        assert isinstance(massif_names, list)
        for massif_name in massif_names:
            self._plot_maxima_time_series(massif_name, show=show)

    def _plot_maxima_time_series(self, massif_name, show=False):
        ax = plt.gca()
        linewidth = 5
        x = self.study.ordered_years
        for altitude, study in self.altitude_to_study.items():
            y = study.massif_name_to_annual_maxima[massif_name]
            label = '{} m'.format(altitude)
            ax.plot(x, y, linewidth=linewidth, label=label)
        ax.xaxis.set_ticks(x[1::10])
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.legend()
        plot_name = 'Annual maxima of {} in {}'.format(SCM_STUDY_CLASS_TO_ABBREVIATION[self.study_class], massif_name)
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('years', fontsize=15)
        self.show_or_save_to_file(plot_name=plot_name, show=show)
        ax.clear()
