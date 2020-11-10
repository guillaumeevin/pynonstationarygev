import matplotlib.pyplot as plt
from itertools import chain

import numpy as np
from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import altitudes_for_groups
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.snowfall_plot import \
    fit_linear_regression
from projects.exceeding_snow_loads.utils import paper_altitudes


class PointwiseGevStudyVisualizer(AltitudesStudies):

    def __init__(self, study_class, altitudes, spatial_transformation_class=None, temporal_transformation_class=None,
                 **kwargs_study):
        super().__init__(study_class, altitudes, spatial_transformation_class, temporal_transformation_class,
                         **kwargs_study)
        # self.altitudes_for_temporal_hypothesis = [min(self.altitudes), 2100, max(self.altitudes)]
        self.altitudes_for_temporal_hypothesis = [600, 1500, 2400, 3300]

    def plot_gev_params_against_altitude(self):
        legend = False
        param_names = GevParams.PARAM_NAMES + [100]
        if legend:
            param_names = param_names[:1]
        for j, param_name in enumerate(param_names):
            ax = plt.gca()

            massif_name_to_linear_coef = {}
            massif_name_to_r2_score = {}
            massif_names = self.study.all_massif_names()[:]
            for i in range(8):
                for massif_name in massif_names[i::8]:
                    linear_coef, _, r2 = self._plot_gev_params_against_altitude_one_massif(ax, massif_name, param_name)
                    massif_name_to_linear_coef[massif_name] = 100 * linear_coef[0]
                    massif_name_to_r2_score[massif_name] = str(round(r2, 2))
            print(param_name, np.mean([c for c in massif_name_to_linear_coef.values()]))

            # Display x label
            xticks = [1000 * i for i in range(1, 4)]
            ax.set_xticks(xticks)
            fontsize_label = 15
            ax.tick_params(labelsize=fontsize_label)

            # Compute the y label
            if param_name in GevParams.PARAM_NAMES:
                ylabel = GevParams.full_name_from_param_name(param_name) + ' parameter'
            else:
                ylabel = '{}-year return levels'.format(param_name)
            # Add units
            if param_name == GevParams.SHAPE:
                unit = 'no unit'
            else:
                unit = self.study.variable_unit
            ylabel += ' ({})'.format(unit)

            # Display the y label on the twin axis
            if param_name in [100, GevParams.SCALE]:
                ax2 = ax.twinx()
                ax2.set_yticks(ax.get_yticks())
                ax2.set_ylim(ax.get_ylim())
                ax2.set_ylabel(ylabel, fontsize=fontsize_label)
                ax2.tick_params(labelsize=fontsize_label)
                ax.set_yticks([])
                tight_layout = False
            else:
                ax.tick_params(labelsize=fontsize_label)
                tight_layout = True
                ax.set_ylabel(ylabel, fontsize=fontsize_label)
            # Make room for the ylabel
            if param_name == 100:
                plt.gcf().subplots_adjust(right=0.85)

            plot_name = '{} change with altitude'.format(param_name)

            # # Display the legend
            if legend:
                # ax.legend(labelspacing=2.5, ncol=8, handlelength=12, markerscale=0.7, bbox_to_anchor=(1.05, 1), loc='upper left',
                #           prop={'size': 2}, fontsize='x-large')
                ax.legend(labelspacing=2.5, ncol=8, handlelength=10, markerscale=0.7, bbox_to_anchor=(1.05, 1), loc='upper left',
                          prop={'size': 2}, fontsize='xx-large')
                plt.gcf().subplots_adjust(right=0.15)
                ax.set_yticks([])
                ax.set_ylabel('')

            # plt.show()
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=tight_layout, show=False)
            ax.clear()
            plt.close()

            # Plot map of slope for each massif
            visualizer = StudyVisualizer(study=self.study, show=False, save_to_file=True)
            idx = 8 if param_name == GevParams.SHAPE else 1
            the = ' the' if param_name in GevParams.PARAM_NAMES else ''
            label = 'Elevation gradient for\n{} {}'.format(the, ylabel[:-idx] + '/100m)')
            gev_param_name_to_graduation = {
                GevParams.LOC: 0.5,
                GevParams.SCALE: 0.1,
                GevParams.SHAPE: 0.01,
                100: 1,
            }
            if param_name == GevParams.SHAPE:
                print(massif_name_to_linear_coef)
            visualizer.plot_map(cmap=plt.cm.coolwarm,
                                graduation=gev_param_name_to_graduation[param_name],
                                label=label, massif_name_to_value=massif_name_to_linear_coef,
                                plot_name=label.replace('/', ' every '), add_x_label=False,
                                negative_and_positive_values=param_name == GevParams.SHAPE,
                                add_colorbar=True,
                                massif_name_to_text=massif_name_to_r2_score,
                                fontsize_label=13,
                                )
            plt.close()

    def _plot_gev_params_against_altitude_one_massif(self, ax, massif_name, param_name):
        altitudes = []
        params = []
        # confidence_intervals = []
        for altitude, study in self.altitude_to_study.items():
            if massif_name in study.massif_name_to_stationary_gev_params:
                gev_params = study.massif_name_to_stationary_gev_params[massif_name]
                altitudes.append(altitude)
                if param_name in GevParams.PARAM_NAMES:
                    param = gev_params.to_dict()[param_name]
                else:
                    assert isinstance(param_name, int)
                    param = gev_params.return_level(return_period=param_name)
                params.append(param)
                # confidence_intervals.append(gev_params.param_name_to_confidence_interval[param_name])
        massif_id = self.study.all_massif_names().index(massif_name)
        plot_against_altitude(altitudes, ax, massif_id, massif_name, params, fill=False)

        return fit_linear_regression(altitudes, params)
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

    def plot_gev_params_against_time_for_all_altitudes(self):
        for altitude in self.altitudes_for_temporal_hypothesis:
            self._plot_gev_params_against_time_for_one_altitude(altitude)

    def _plot_gev_params_against_time_for_one_altitude(self, altitude):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for massif_name in self.study.all_massif_names()[:]:
                self._plot_gev_params_against_time_for_one_altitude_and_one_massif(ax, massif_name, param_name,
                                                                                   altitude,
                                                                                   massif_name_as_labels=True)
            ax.legend(prop={'size': 7}, ncol=3)
            ax.set_xlabel('Year')
            ax.set_ylabel(param_name + ' for altitude={}'.format(altitude))
            xlabels = ['-'.join([str(e) for e in t]) for t in self.year_min_and_max_list]
            ax.set_xticks(self.min_years_for_plot_x_axis)
            ax.set_xticklabels(xlabels)
            # ax.tick_params(labelsize=5)
            plot_name = '{} change /all with years /for altitude={}'.format(param_name, altitude)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    def _plot_gev_params_against_time_for_one_altitude_and_one_massif(self, ax, massif_name, param_name, altitude,
                                                                      massif_name_as_labels):
        study = self.altitude_to_study[altitude]
        if massif_name in study.study_massif_names:
            gev_params_list = study.massif_name_to_gev_param_list(self.year_min_and_max_list)[massif_name]
            params = [gev_params.to_dict()[param_name] for gev_params in gev_params_list]
            # params = np.array(params)
            # param_normalized = params / np.sqrt(np.sum(np.power(params, 2)))
            # confidence_intervals = [gev_params.param_name_to_confidence_interval[param_name] for gev_params in
            #                         gev_params_list]
            massif_id = self.study.all_massif_names().index(massif_name)
            plot_against_altitude(self.min_years_for_plot_x_axis, ax, massif_id, massif_name, params,
                                  altitude, False,
                                  massif_name_as_labels)
            # plot_against_altitude(self.years, ax, massif_id, massif_name, confidence_intervals, True)

    # plot for each massif against the time

    def plot_gev_params_against_time_for_all_massifs(self):
        for massif_name in self.study.all_massif_names():
            self._plot_gev_params_against_time_for_one_massif(massif_name)

    def _plot_gev_params_against_time_for_one_massif(self, massif_name):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for altitude in self.altitudes_for_temporal_hypothesis:
                self._plot_gev_params_against_time_for_one_altitude_and_one_massif(ax, massif_name, param_name,
                                                                                   altitude,
                                                                                   massif_name_as_labels=False)
            ax.legend()
            ax.set_xlabel('Year')
            ax.set_ylabel(param_name + ' for {}'.format(massif_name))
            xlabels = ['-'.join([str(e) for e in t]) for t in self.year_min_and_max_list]
            ax.set_xticks(self.min_years_for_plot_x_axis)
            ax.set_xticklabels(xlabels)
            plot_name = '{} change /with years /for {}'.format(param_name, massif_name)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    # PLot for each massif the derivative against the time for each altitude

    def plot_time_derivative_against_time(self):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for massif_name in self.study.all_massif_names()[:]:
                self._plot_gev_params_time_derivative_against_altitude_one_massif(ax, massif_name, param_name)
            ax.legend(prop={'size': 7}, ncol=3)
            ax.set_xlabel('Altitude')
            ax.set_ylabel(param_name)
            plot_name = '{} change /time derivative with altitude'.format(param_name)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    def _plot_gev_params_time_derivative_against_altitude_one_massif(self, ax, massif_name, param_name):
        altitudes = []
        time_derivatives = []
        for altitude, study in self.altitude_to_study.items():
            if (massif_name in study.study_massif_names) and ("Mercan" not in massif_name):
                gev_params_list = study.massif_name_to_gev_param_list(self.year_min_and_max_list)[massif_name]
                params = [gev_params.to_dict()[param_name] for gev_params in gev_params_list]
                x = list(range(len(params)))
                y = params
                a = self.get_robust_slope(x, y)
                time_derivatives.append(a)
                altitudes.append(altitude)
        massif_id = self.study.all_massif_names().index(massif_name)
        plot_against_altitude(altitudes, ax, massif_id, massif_name, time_derivatives, fill=False)

    def get_robust_slope(self, x, y):
        a, *_ = fit_linear_regression(x=x, y=y)
        a_list = [a]
        for i in range(len(x)):
            x_copy, y_copy = x[:], y[:]
            x_copy.pop(i)
            y_copy.pop(i)
            a, *_ = fit_linear_regression(x=x_copy, y=y_copy)
            a_list.append(a)
        return np.mean(np.array(a_list))


if __name__ == '__main__':
    altitudes = list(chain.from_iterable(altitudes_for_groups))

    # altitudes = paper_altitudes
    # altitudes = [1800, 2100]
    visualizer = PointwiseGevStudyVisualizer(SafranSnowfall1Day, altitudes=altitudes)
    visualizer.plot_gev_params_against_altitude()
    # visualizer.plot_gev_params_against_time_for_all_altitudes()
    # visualizer.plot_gev_params_against_time_for_all_massifs()
    # visualizer.plot_time_derivative_against_time()
