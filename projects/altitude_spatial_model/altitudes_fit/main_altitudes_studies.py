import time
from typing import List

import matplotlib as mpl

from projects.altitude_spatial_model.altitudes_fit.plot_histogram_altitude_studies import \
    plot_histogram_all_trends_against_altitudes, plot_histogram_all_models_against_altitudes

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from projects.altitude_spatial_model.altitudes_fit.utils_altitude_studies_visualizer import load_visualizer_list

from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import altitudes_for_groups
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.plot_total_aic import plot_individual_aic

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days, SafranPrecipitation1Day, SafranPrecipitation3Days, \
    SafranPrecipitation5Days, SafranPrecipitation7Days
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels


def main():
    study_classes = [SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, SafranSnowfall7Days][:1]
    # study_classes = [SafranPrecipitation1Day][:1]
    seasons = [Season.annual, Season.winter, Season.spring, Season.automn][:1]

    fast = False
    if fast is None:
        massif_names = None
        altitudes_list = altitudes_for_groups[:2]
    elif fast:
        massif_names = ['Mercantour', 'Vercors', 'Ubaye']
        altitudes_list = altitudes_for_groups[:2]
    else:
        massif_names = None
        altitudes_list = altitudes_for_groups

    main_loop(altitudes_list, massif_names, seasons, study_classes)


def main_loop(altitudes_list, massif_names, seasons, study_classes):
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    for season in seasons:
        for study_class in study_classes:
            print('Inner loop', season, study_class)
            visualizer_list = load_visualizer_list(season, study_class, altitudes_list, massif_names)
            plot_visualizers(massif_names, visualizer_list)
            for visualizer in visualizer_list:
                plot_visualizer(massif_names, visualizer)
            del visualizer_list
            time.sleep(2)


def plot_visualizers(massif_names, visualizer_list):
    plot_histogram_all_trends_against_altitudes(massif_names, visualizer_list)
    plot_histogram_all_models_against_altitudes(massif_names, visualizer_list)


def plot_visualizer(massif_names, visualizer):
    # Plot time series
    visualizer.studies.plot_maxima_time_series(massif_names=massif_names)
    # Plot moments against altitude
    # for std in [True, False][:]:
    #     for change in [True, False, None]:
    #         studies.plot_mean_maxima_against_altitude(massif_names=massif_names, std=std, change=change)
    # Plot the results for the model that minimizes the individual aic
    plot_individual_aic(visualizer)
    # Plot the results for the model that minimizes the total aic
    # plot_total_aic(model_classes, visualizer)


if __name__ == '__main__':
    main()
