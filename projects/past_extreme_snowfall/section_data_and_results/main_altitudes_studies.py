import datetime
import time
from typing import List
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import PercentFormatter

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from root_utils import get_display_name_from_object_type

from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020, \
    SafranSnowfall2019
from extreme_trend.one_fold_fit.utils_altitude_studies_visualizer import load_visualizer_list

from extreme_trend.one_fold_fit.plots.plot_histogram_altitude_studies import \
    plot_shoe_plot_changes_against_altitude, plot_histogram_all_trends_against_altitudes, \
    plot_histogram_all_models_against_altitudes

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel

from extreme_fit.model.utils import set_seed_for_test
from extreme_trend.one_fold_fit.plots.plot_coherence_curves import plot_coherence_curves

from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days, SafranSnowfallCenterOnDay1day, SafranSnowfallNotCenterOnDay1day, \
    SafranSnowfallCenterOnDay1dayMeanRate, SafranPrecipitation1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    study_classes = [SafranSnowfall1Day
                        , SafranSnowfall3Days,
                     SafranSnowfall5Days, SafranSnowfall7Days][:1]
    # study_classes = [SafranSnowfall2020, SafranSnowfall2019, SafranSnowfallCenterOnDay1day,
    #                  SafranSnowfallNotCenterOnDay1day,
    #                  SafranSnowfallCenterOnDay1dayMeanRate, SafranSnowfall1Day][:1]
    # study_classes = [SafranSnowfallNotCenterOnDay1day, SafranSnowfall2019]
    seasons = [Season.annual, Season.winter, Season.spring, Season.automn][:1]

    # study_classes = [SafranPrecipitation1Day]
    # seasons = [Season.winter, Season.automn][:1]

    set_seed_for_test()
    model_must_pass_the_test = False
    AbstractExtractEurocodeReturnLevel.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.2

    fast = None
    if fast is None:
        massif_names = None
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        altitudes_list = altitudes_for_groups[:1]
        # altitudes_list = altitudes_for_groups[1:2]
    elif fast:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = ['Vanoise', 'Haute-Maurienne', 'Vercors'][:1]
        altitudes_list = altitudes_for_groups[2:3]
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
            print('Run', get_display_name_from_object_type(study_class), season)
            visualizer_list = load_visualizer_list(season, study_class, altitudes_list, massif_names,
                                                   model_must_pass_the_test)
            with_significance = False
            plot_visualizers(massif_names, visualizer_list, with_significance)
            for visualizer in visualizer_list:
                plot_visualizer(massif_names, visualizer, with_significance)
            del visualizer_list
            time.sleep(2)


def plot_visualizers(massif_names, visualizer_list, with_significance):
    # return_level_plots(massif_names, visualizer_list, with_significance)
    # qqplot_plots(visualizer_list)

    # plot_histogram_all_models_against_altitudes(massif_names, visualizer_list)
    # plot_shoe_plot_ratio_interval_size_against_altitude(massif_names, visualizer_list)
    # for relative in [True, False]:
    #     plot_shoe_plot_changes_against_altitude(massif_names, visualizer_list, relative=relative, with_significance=with_significance)
    # plot_coherence_curves(['Vanoise'], visualizer_list)
    pass


def plot_visualizer(massif_names, visualizer, with_significance):
    # Plot time series
    # visualizer.studies.plot_maxima_time_series(massif_names)
    # visualizer.studies.plot_maxima_time_series(['Vanoise'])

    # visualizer.plot_shape_map()
    visualizer.plot_moments(with_significance)

    # for std in [True, False]:
    #     visualizer.studies.plot_mean_maxima_against_altitude(std=std)
    pass


def qqplot_plots(visualiser_list):
    metric_list = []
    for visualizer in visualiser_list:
        metric_list.extend(visualizer.plot_qqplots())
    # Create an histogram for the metric
    ax = plt.gca()
    count_above_5_percent = [int(m >= 0.05) for m in metric_list]
    percentage_above_5_percent = 100 * sum(count_above_5_percent) / len(count_above_5_percent)
    print("Percentage above 5 percent", percentage_above_5_percent)
    ax.hist(metric_list, bins=20, range=[0, 1], weights=np.ones(len(metric_list)) / len(metric_list))
    ax.set_xlim((0, 1))
    ylim = ax.get_ylim()
    ax.vlines(0.05, ymin=ylim[0], ymax=ylim[1], color='k', linestyles='dashed', label='0.05 significance level')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('p-value for the Anderson-Darling test')
    ax.set_ylabel('Percentage')
    ax.legend()
    visualizer = visualiser_list[0]
    visualizer.plot_name = 'All pvalues'
    visualizer.show_or_save_to_file()


def return_level_plots(massif_names, visualizer_list, with_significance):
    default_return_period = OneFoldFit.return_period
    for return_period in [10, 100]:
        OneFoldFit.return_period = return_period
        plot_histogram_all_trends_against_altitudes(massif_names, visualizer_list, with_significance=with_significance)
    OneFoldFit.return_period = default_return_period


if __name__ == '__main__':
    main()
