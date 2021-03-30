import matplotlib.pyplot as plt
import os.path as op
from itertools import chain

import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import _gcm_rcm_couple_adamont_v2_to_full_name
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import get_gcm_rcm_couples, rcp_scenarios
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.utils import fit_linear_regression
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups


class PointwiseGevStudyVisualizer(AltitudesStudies):

    def __init__(self, study_class, altitudes, spatial_transformation_class=None, temporal_transformation_class=None,
                 **kwargs_study):
        super().__init__(study_class, altitudes, spatial_transformation_class, temporal_transformation_class,
                         **kwargs_study)

    def plot_gev_params_against_altitude(self):
        legend = False
        elevation_as_xaxis = False
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
                    linear_coef, _, r2 = self._plot_gev_params_against_altitude_one_massif(ax, massif_name, param_name,
                                                                                           elevation_as_xaxis,
                                                                                           legend=legend)
                    massif_name_to_linear_coef[massif_name] = 100 * linear_coef[0]
                    massif_name_to_r2_score[massif_name] = str(round(r2, 2))
            print(param_name, np.mean([c for c in massif_name_to_linear_coef.values()]))

            # Display x label
            elevation_ticks = [500 * i for i in range(1, 8)]
            if elevation_as_xaxis:
                ax.set_xticks(elevation_ticks)
            else:
                ax.set_yticks(elevation_ticks)
                if j == 2:
                    ax.set_xlim(right=0.45)
            fontsize_label = 15
            ax.tick_params(labelsize=fontsize_label)

            # Compute the y label
            if param_name in GevParams.PARAM_NAMES:
                value_label = GevParams.full_name_from_param_name(param_name) + ' parameter'
            else:
                value_label = '{}-year return levels'.format(param_name)
            # Add units
            if param_name == GevParams.SHAPE:
                unit = 'no unit'
            else:
                unit = self.study.variable_unit
            value_label += ' ({})'.format(unit)
            value_label = value_label.capitalize()

            if elevation_as_xaxis:
                # Display the y label on the twin axis
                if param_name in [100, GevParams.SCALE]:
                    ax2 = ax.twinx()
                    ax2.set_yticks(ax.get_yticks())
                    ax2.set_ylim(ax.get_ylim())
                    ax2.set_ylabel(value_label, fontsize=fontsize_label)
                    ax2.tick_params(labelsize=fontsize_label)
                    ax.set_yticks([])
                    tight_layout = False
                else:
                    ax.tick_params(labelsize=fontsize_label)
                    tight_layout = True
                    ax.set_ylabel(value_label, fontsize=fontsize_label)
                # Make room for the ylabel
                if param_name == 100:
                    plt.gcf().subplots_adjust(right=0.85)
            else:
                if param_name in [GevParams.LOC, GevParams.SCALE]:
                    ax2 = ax.twiny()
                    ax2.set_xticks(ax.get_xticks())
                    ax.set_xticks([])
                else:
                    ax2 = ax

                ax2.set_xlim(ax.get_xlim())
                ax2.set_xlabel(value_label, fontsize=fontsize_label)
                ax2.tick_params(labelsize=fontsize_label)

                if param_name in [100, GevParams.SCALE]:
                    ax3 = ax2.twinx()
                    ax3.set_yticks(ax2.get_yticks())
                    ax3.set_ylim(ax2.get_ylim())
                    ax3.tick_params(labelsize=fontsize_label)
                    ax2.set_yticks([])
                    ax3.set_ylabel('Elevation (m)', fontsize=fontsize_label)
                else:
                    ax.set_ylabel('Elevation (m)', fontsize=fontsize_label)

                tight_layout = False

            plot_name = '{} change with altitude'.format(param_name)
            if isinstance(self.study, AbstractAdamontStudy):
                plot_name = op.join(plot_name, _gcm_rcm_couple_adamont_v2_to_full_name[self.study.gcm_rcm_couple])

            # # Display the legend
            if legend:
                # ax.legend(labelspacing=2.5, ncol=8, handlelength=12, markerscale=0.7, bbox_to_anchor=(1.05, 1), loc='upper left',
                #           prop={'size': 2}, fontsize='x-large')
                # ax.legend(labelspacing=1, ncol=8, handlelength=5, bbox_to_anchor=(1.05, 1), loc='upper left',
                #           prop={'size': 4}, fontsize='xx-large', columnspacing=0.5)
                ax.legend(ncol=8, bbox_to_anchor=(1.05, 1), loc='upper left',
                          prop={'size': 3.5}, handlelength=5, fontsize='xx-large', columnspacing=0.5,
                          handletextpad=0.5)

                # handles, labels = ax.get_legend_handles_labels()
                # print(type(handles))
                # handles = np.array(handles).reshape((3, 8)).transpose().flatten()
                # labels = np.array(handles).reshape((3, 8)).transpose().flatten()
                # ax.legend(handles, labels)

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
            label = 'Elevation gradient for\n{} {}'.format(the, value_label[:-idx] + '/100m)')
            plot_name = label.replace('/', ' every ')
            if isinstance(self.study, AbstractAdamontStudy):
                plot_name = op.join(plot_name, _gcm_rcm_couple_adamont_v2_to_full_name[self.study.gcm_rcm_couple])

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
                                plot_name=plot_name, add_x_label=False,
                                negative_and_positive_values=param_name == GevParams.SHAPE,
                                add_colorbar=True,
                                massif_name_to_text=massif_name_to_r2_score,
                                fontsize_label=13,
                                )
            plt.close()

    def _plot_gev_params_against_altitude_one_massif(self, ax, massif_name, param_name, elevation_as_xaxis,
                                                     legend=False):
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
        massif_id = self.study.all_massif_names().index(massif_name)
        plot_against_altitude(altitudes, ax, massif_id, massif_name, params, fill=False,
                              elevation_as_xaxis=elevation_as_xaxis,
                              legend=legend)

        return fit_linear_regression(altitudes, params)


def main_paper2():
    altitudes = list(chain.from_iterable(altitudes_for_groups))

    # altitudes = paper_altitudes
    altitudes = [1800, 2100]
    visualizer = PointwiseGevStudyVisualizer(SafranSnowfall1Day, altitudes=altitudes)
    visualizer.plot_gev_params_against_altitude()

    # visualizer.plot_gev_params_against_time_for_all_altitudes()
    # visualizer.plot_gev_params_against_time_for_all_massifs()
    # visualizer.plot_time_derivative_against_time()


def main_paper3():
    altitudes = list(chain.from_iterable(altitudes_for_groups))
    # altitudes = [1200, 1500, 1800]
    for scenario in rcp_scenarios[:]:
        gcm_rcm_couples = get_gcm_rcm_couples(scenario)
        # gcm_rcm_couples =[('CNRM-CM5', 'CCLM4-8-17')]
        for gcm_rcm_couple in gcm_rcm_couples:
            visualizer = PointwiseGevStudyVisualizer(AdamontSnowfall, altitudes=altitudes, scenario=scenario,
                                                     gcm_rcm_couple=gcm_rcm_couple)
            visualizer.plot_gev_params_against_altitude()


if __name__ == '__main__':
    main_paper3()
