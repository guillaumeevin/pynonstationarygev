import matplotlib.pyplot as plt
from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.exceeding_snow_loads.utils import paper_altitudes


class PointwiseGevStudyVisualizer(AltitudesStudies):

    def plot_gev_params_against_altitude(self):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for massif_name in self.study.all_massif_names()[:]:
                self._plot_gev_params_against_altitude_one_massif(ax, massif_name, param_name)
            ax.legend(prop={'size': 7}, ncol=3)
            ax.set_xlabel('Altitude')
            ax.set_ylabel(param_name)
            plot_name = '{} change /with altitude'.format(param_name)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    def _plot_gev_params_against_altitude_one_massif(self, ax, massif_name, param_name):
        altitudes = []
        params = []
        confidence_intervals = []
        for altitude, study in self.altitude_to_study.items():
            if massif_name in study.study_massif_names:
                altitudes.append(altitude)
                gev_params = study.massif_name_to_stationary_gev_params[massif_name]
                params.append(gev_params.to_dict()[param_name])
                confidence_intervals.append(gev_params.param_name_to_confidence_interval[param_name])
        massif_id = self.study.all_massif_names().index(massif_name)
        plot_against_altitude(altitudes, ax, massif_id, massif_name, params, fill=False)
        # plot_against_altitude(altitudes, ax, massif_id, massif_name, confidence_intervals, fill=True)

    # Plot against the time

    @property
    def year_min_and_max_list(self):
        l = []
        year_min, year_max = 1959, 1989
        for shift in range(0, 7):
            l.append((year_min + 5 * shift, year_max + 5 * shift))
        return l

    @property
    def min_years_for_plot_x_axis(self):
        return [c[0] for c in self.year_min_and_max_list]

    def plot_gev_params_against_time(self):
        for altitude in self.altitudes[:]:
            self._plot_gev_params_against_time(altitude)

    def _plot_gev_params_against_time(self, altitude):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for massif_name in self.study.all_massif_names()[:]:
                self._plot_gev_params_against_time_one_massif(ax, massif_name, param_name, altitude)
            ax.legend(prop={'size': 7}, ncol=3)
            ax.set_xlabel('Year')
            ax.set_ylabel(param_name + ' for altitude={}'.format(altitude))
            xlabels = ['-'.join([str(e) for e in t]) for t in self.year_min_and_max_list]
            ax.set_xticks(self.min_years_for_plot_x_axis)
            ax.set_xticklabels(xlabels)
            # ax.tick_params(labelsize=5)
            plot_name = '{} change /with years for altitude={}'.format(param_name, altitude)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    def _plot_gev_params_against_time_one_massif(self, ax, massif_name, param_name, altitude):
        study = self.altitude_to_study[altitude]
        if massif_name in study.study_massif_names:
            gev_params_list = study.massif_name_to_gev_param_list(self.year_min_and_max_list)[massif_name]
            params = [gev_params.to_dict()[param_name] for gev_params in gev_params_list]
            confidence_intervals = [gev_params.param_name_to_confidence_interval[param_name] for gev_params in
                                    gev_params_list]
            massif_id = self.study.all_massif_names().index(massif_name)
            plot_against_altitude(self.min_years_for_plot_x_axis, ax, massif_id, massif_name, params, fill=False)
            # plot_against_altitude(self.years, ax, massif_id, massif_name, confidence_intervals, fill=True)


if __name__ == '__main__':
    altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
    # altitudes = paper_altitudes
    # altitudes = [1800, 2100]
    visualizer = PointwiseGevStudyVisualizer(SafranSnowfall1Day, altitudes=altitudes)
    visualizer.plot_gev_params_against_altitude()
    visualizer.plot_gev_params_against_time()
