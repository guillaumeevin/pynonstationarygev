import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
import matplotlib.pyplot as plt

from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels


def parameter_plot(visualizer: AltitudesStudiesVisualizerForNonStationaryModels):
    for order in GevParams.PARAM_NAMES:
        _parameter_plot(visualizer, order)

def _parameter_plot(visualizer: AltitudesStudiesVisualizerForNonStationaryModels, order):
    covariates = np.linspace(1.5, 4, num=50)
    altitude = visualizer.altitude_group.reference_altitude
    print(altitude, order)
    ax = plt.gca()
    for massif_name, one_fold_fit in visualizer.massif_name_to_one_fold_fit.items():
        parameter_values = [one_fold_fit.get_moment(altitude, c, order) for c in covariates]
        ax.plot(covariates, parameter_values)

    visualizer.plot_name = 'parameter plot for {}'.format(order)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()

def get_parameter_values_list(visualizer: AltitudesStudiesVisualizerForNonStationaryModels, covariates):
    parameter_values_list = []
    for massif_name, one_fold_fit in visualizer.massif_name_to_one_fold_fit.items():
        parameter_values = [one_fold_fit.get_moment(visualizer.altitude_group.reference_altitude, c, GevParams.SHAPE)
                            for c in covariates]
        parameter_values_list.append(parameter_values)
    return parameter_values_list

def shape_parameter_plot(visualizer, covariates, parameter_values_list):
    ax = plt.gca()

    for parameter_values in parameter_values_list:
        ax.plot(covariates, parameter_values, color='grey')
    legend_fontsize = 16
    ticksize = 14
    covariates_to_show = [1.5, 2, 2.5, 3, 3.5, 4]
    ax.set_xlim((covariates_to_show[0], covariates_to_show[-1]))
    ax.set_xticks(covariates_to_show)
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates_to_show], fontsize=ticksize)
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    ax.set_ylabel('Shape parameter (-)', fontsize=legend_fontsize)
    visualizer.plot_name = 'parameter plot for the shape'
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()
