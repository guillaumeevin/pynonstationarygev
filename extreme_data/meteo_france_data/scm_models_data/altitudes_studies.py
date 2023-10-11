import pandas as pd
import numpy as np
from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import scenario_to_str, gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION, STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from root_utils import memoize
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.spatio_temporal_coordinates_for_climate_models import \
    SpatioTemporalCoordinatesForClimateModels
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
import matplotlib.pyplot as plt


class AltitudesStudies(object):

    def __init__(self, study_class, altitudes,
                 **kwargs_study):
        self.study_class = study_class
        self.altitudes = altitudes
        self.altitude_to_study = OrderedDict()  # type: OrderedDict[int, AbstractStudy]
        for altitude in self.altitudes:
            study = study_class(altitude=altitude, **kwargs_study)
            self.altitude_to_study[altitude] = study

    @cached_property
    def study(self) -> AbstractStudy:
        return list(self.altitude_to_study.values())[0]

    # Dataset Loader

    @memoize
    def spatio_temporal_dataset_memoize(self, massif_name, massif_altitude):
        return self.spatio_temporal_dataset(massif_name, [massif_altitude])

    def spatio_temporal_dataset(self, massif_name, massif_altitudes=None,
                                gcm_rcm_couple_as_pseudo_truth=None):
        assert len(massif_name) > 1, massif_name
        coordinate_values_to_maxima = {}
        if massif_altitudes is None:
            massif_altitudes = self.massif_name_to_altitudes[massif_name]
        if len(massif_altitudes) == 0:
            print('{} has no data for these altitudes: {}'.format(massif_name, self.altitudes))
        for altitude in massif_altitudes:
            study = self.altitude_to_study[altitude]
            for year, maxima in zip(study.ordered_years, study.massif_name_to_annual_maxima[massif_name]):
                # Cast to float
                year, altitude = float(year), float(altitude)
                if len(massif_altitudes) == 1:
                    coordinate_values = [year]
                else:
                    coordinate_values = [altitude, year]
                coordinate_values_to_maxima[tuple(coordinate_values)] = [maxima]

        coordinates = self.spatio_temporal_coordinates(massif_altitudes)
        # Remove the spatial coordinate if we only have one altitude
        if len(massif_altitudes) == 1:
            df = pd.concat([coordinates.df_temporal_coordinates(), coordinates.df_coordinate_climate_model], axis=1)
            coordinates = AbstractTemporalCoordinates.from_df(df)

        observations = AnnualMaxima.from_coordinates(coordinates, coordinate_values_to_maxima)
        coordinates.gcm_rcm_couple_as_pseudo_truth = gcm_rcm_couple_as_pseudo_truth
        return AbstractDataset(observations=observations, coordinates=coordinates)

    @cached_property
    def massif_name_to_altitudes(self):
        massif_names = self.study.all_massif_names()
        massif_name_to_altitudes = {massif_name: [] for massif_name in massif_names}
        for altitude in self.altitudes:
            study = self.altitude_to_study[altitude]
            for massif_name in study.study_massif_names:
                massif_name_to_altitudes[massif_name].append(altitude)
        return massif_name_to_altitudes

    # Coordinates Loader

    def spatio_temporal_coordinates(self, massif_altitudes=None):
        if massif_altitudes is None or set(massif_altitudes) == set(self.altitudes):
            spatial_coordinates = self.spatial_coordinates
        else:
            assert len(massif_altitudes) > 0
            spatial_coordinates = self.spatial_coordinates_for_altitudes(massif_altitudes)
        if isinstance(self.study, AbstractAdamontStudy):
            return SpatioTemporalCoordinatesForClimateModels(spatial_coordinates=spatial_coordinates,
                                                             temporal_coordinates=self.temporal_coordinates,
                                                             gcm_rcm_couple=self.study.gcm_rcm_couple,
                                                             scenario_str=scenario_to_str(self.study.scenario),
                                                             )
        else:
            return SpatioTemporalCoordinatesForClimateModels(spatial_coordinates=spatial_coordinates,
                                                             temporal_coordinates=self.temporal_coordinates,
                                                             gcm_rcm_couple=(np.nan, np.nan),
                                                             scenario_str=np.nan,
                                                             )

    @cached_property
    def temporal_coordinates(self):
        return ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=self.study.nb_years,
                                                                     start=self.study.year_min)

    @cached_property
    def spatial_coordinates(self):
        return self.spatial_coordinates_for_altitudes(self.altitudes)

    def spatial_coordinates_for_altitudes(self, altitudes):
        return AbstractSpatialCoordinates.from_list_x_coordinates(x_coordinates=altitudes)

    @cached_property
    def _df_coordinates(self):
        return AbstractSpatioTemporalCoordinates.get_df_from_spatial_and_temporal_coordinates(self.spatial_coordinates,
                                                                                              self.temporal_coordinates)

    # Some visualization

    def show_or_save_to_file(self, plot_name, show=False, no_title=False, tight_layout=None):
        study_visualizer = StudyVisualizer(study=self.study, show=show, save_to_file=not show)
        study_visualizer.plot_name = plot_name
        study_visualizer.show_or_save_to_file(add_classic_title=False, dpi=500, no_title=no_title,
                                              tight_layout=tight_layout)

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
        for altitude, study in list(self.altitude_to_study.items()):
            if massif_name in study.massif_name_to_annual_maxima:
                y = study.massif_name_to_annual_maxima[massif_name]
                label = '{} m'.format(altitude)
                ax.plot(x, y, linewidth=2, label=label)
        ax.xaxis.set_ticks([e for e in x if e % 10 == 0][::2])
        ax.set_xlim((x[0], x[-1]))

        # Plot for the paper 2
        if massif_name == "Vanoise" and (not issubclass(self.study_class, AbstractAdamontStudy)):
            # ax.yaxis.set_ticks([25 * (j) for j in range(6)])
            ax.yaxis.set_ticks([25 * (j) for j in range(7)])
            labelsize = 20
            fontsize = 15
        else:
            fontsize = 10
            labelsize = 10

        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], prop={'size': labelsize})
        plot_name = 'Annual maxima of {} in {}'.format(STUDY_CLASS_TO_ABBREVIATION[self.study_class],
                                                       massif_name.replace('_', ' '))
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=fontsize)
        # ax.set_xlabel('years', fontsize=15)
        plot_name = 'time series/' + plot_name
        self.show_or_save_to_file(plot_name=plot_name, show=show, no_title=True, tight_layout=True)
        ax.clear()
        plt.close()

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
        # if change is False:
        # moment += ' (for the 60 years of data)'
        plot_name = '{} of {} maxima of {}'.format(moment, self.study.season_name,
                                                   SCM_STUDY_CLASS_TO_ABBREVIATION[self.study_class])
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('altitudes', fontsize=15)
        lim_down, lim_up = ax.get_ylim()
        lim_up += (lim_up - lim_down) / 3
        ax.set_ylim([lim_down, lim_up])
        ax.tick_params(axis='both', which='major', labelsize=13)
        self.show_or_save_to_file(plot_name=plot_name, show=show, no_title=True)
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
