import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from extreme_data.meteo_france_data.scm_models_data.safran.gap_between_study import GapBetweenSafranSnowfall2019And2020, \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019Recentered, GapBetweenSafranSnowfall2019AndMySafranSnowfall2019, \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019NotRecentered, \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019RecenteredMeanRate
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    STUDY_CLASS_TO_ABBREVIATION
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies


def comparison_plot(altitude_studies: AltitudesStudies, massif_name):
    ax = plt.gca()
    altitudes = [a for a, s in altitude_studies.altitude_to_study.items() if massif_name in s.study_massif_names]
    annual_maxima = [altitude_studies.altitude_to_study[a].massif_name_to_annual_maxima[massif_name] for a in altitudes]
    min_bias, median_bias, max_bias = [[f(maxima) for maxima in annual_maxima] for f in [np.min, np.median, np.max]]

    color = 'blue'
    ax.plot(median_bias, altitudes, label='Median bias', color=color, marker='o')
    ax.fill_betweenx(altitudes, min_bias, max_bias, label='Range for the bias', alpha=0.2, color=color)
    ax.vlines(0, ymin=altitudes[0], ymax=altitudes[-1], color='k')
    massif_name_str = massif_name.replace('_', '-')
    study_str = STUDY_CLASS_TO_ABBREVIATION[type(altitude_studies.study)]
    plot_name = '{} - Bias for {}'.format(massif_name_str, study_str)
    plot_name += '\nWe consider maxima between {} and {}'.format(altitude_studies.study.year_min,
                                                                 altitude_studies.study.year_max)
    ax.yaxis.set_ticks(altitudes)
    ax.set_ylim(top=altitudes[-1] + 500)
    ax.legend(loc='lower left')
    altitude_studies.show_or_save_to_file(plot_name=plot_name)
    plt.close()


def main_comparaison_plot():
    altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
    for study_class in [GapBetweenSafranSnowfall2019AndMySafranSnowfall2019RecenteredMeanRate,
                        GapBetweenSafranSnowfall2019AndMySafranSnowfall2019NotRecentered,
                        GapBetweenSafranSnowfall2019AndMySafranSnowfall2019Recentered,
                        GapBetweenSafranSnowfall2019AndMySafranSnowfall2019,
                        GapBetweenSafranSnowfall2019And2020][-1:]:
        altitude_studies = AltitudesStudies(study_class=study_class,
                                            altitudes=altitudes)
        for massif_name in altitude_studies.study.all_massif_names():
            comparison_plot(altitude_studies, massif_name)


if __name__ == '__main__':
    main_comparaison_plot()
