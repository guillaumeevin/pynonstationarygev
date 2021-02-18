import datetime
import time
from typing import List

import matplotlib

matplotlib.use('Agg')

from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2019, \
    SafranSnowfall2020
from projects.altitude_spatial_model.altitudes_fit.plots.plot_histogram_altitude_studies import \
    plot_shoe_plot_changes_against_altitude, plot_histogram_all_trends_against_altitudes, \
    plot_shoe_plot_ratio_interval_size_against_altitude, plot_histogram_all_models_against_altitudes

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel

import matplotlib as mpl

from extreme_fit.model.utils import set_seed_for_test
from projects.altitude_spatial_model.altitudes_fit.plots.plot_coherence_curves import plot_coherence_curves

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from projects.altitude_spatial_model.altitudes_fit.utils_altitude_studies_visualizer import load_visualizer_list

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import altitudes_for_groups
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.plot_total_aic import plot_individual_aic

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days, SafranDateFirstSnowfall, SafranPrecipitation1Day, \
    SafranPrecipitation3Days, SafranSnowfallCenterOnDay1dayMeanRate, SafranSnowfallNotCenterOnDay1day
from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    study_classes = [SafranSnowfall1Day
                     , SafranSnowfall3Days,
                     SafranSnowfall5Days, SafranSnowfall7Days][:1]
    seasons = [Season.annual, Season.winter, Season.spring, Season.automn][:1]

    set_seed_for_test()
    model_must_pass_the_test = False
    AbstractExtractEurocodeReturnLevel.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.2

    fast = False
    if fast is None:
        massif_names = None
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        altitudes_list = altitudes_for_groups[2:3]
    elif fast:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = ['Vanoise', 'Haute-Maurienne', 'Vercors'][:1]
        altitudes_list = altitudes_for_groups[1:2]
    else:
        massif_names = None
        altitudes_list = altitudes_for_groups[:]

    start = time.time()
    main_loop(altitudes_list, massif_names, seasons, study_classes, model_must_pass_the_test)
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def main_loop(altitudes_list, massif_names, seasons, study_classes, model_must_pass_the_test):
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    for season in seasons:
        for study_class in study_classes:
            print('Inner loop', season, study_class)
            visualizer_list = load_visualizer_list(season, study_class, altitudes_list, massif_names,
                                                   model_must_pass_the_test
                                                )
            plot_visualizers(massif_names, visualizer_list)
            for visualizer in visualizer_list:
                plot_visualizer(massif_names, visualizer)
            del visualizer_list
            time.sleep(2)


def plot_visualizers(massif_names, visualizer_list):
    # plot_histogram_all_models_against_altitudes(massif_names, visualizer_list)
    plot_histogram_all_trends_against_altitudes(massif_names, visualizer_list)
    # plot_shoe_plot_ratio_interval_size_against_altitude(massif_names, visualizer_list)
    for relative in [True, False]:
        plot_shoe_plot_changes_against_altitude(massif_names, visualizer_list, relative=relative)
    plot_coherence_curves(massif_names, visualizer_list)
    # plot_coherence_curves(['Vanoise'], visualizer_list)
    pass


def plot_visualizer(massif_names, visualizer):
    # Plot time series
    # visualizer.studies.plot_maxima_time_series(massif_names)
    # visualizer.studies.plot_maxima_time_series(['Vanoise'])

    # Plot the results for the model that minimizes the individual aic
    plot_individual_aic(visualizer)

    # Plot the results for the model that minimizes the total aic
    # plot_total_aic(model_classes, visualizer)
    pass


if __name__ == '__main__':
    main()
