import numpy as np
import matplotlib

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, gcm_rcm_couple_to_color, \
    gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020
from extreme_data.meteo_france_data.adamont_data.get_list_gcm_rcm_couples_adamont_v2 import get_year_min_and_year_max_used_to_compute_quantile

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    STUDY_CLASS_TO_ABBREVIATION
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies


def compute_bias_and_display_it(ax,
                                altitude_studies_reanalysis: AltitudesStudies,
                                adamont_altitude_studies: AltitudesStudies,
                                gcm_rcm_couple
                                ):
    bias_in_the_mean_maxima = []
    altitudes = []
    for altitude, study_reanalysis in altitude_studies_reanalysis.altitude_to_study.items():
        altitudes.append(altitude)
        adamont_study = adamont_altitude_studies.altitude_to_study[altitude]
        mean_maxima_adamont = adamont_study.mean_annual_maxima
        mean_maxima_reanalysis = study_reanalysis.mean_annual_maxima
        bias = mean_maxima_adamont - mean_maxima_reanalysis
        bias_in_the_mean_maxima.append(bias)

    color = gcm_rcm_couple_to_color[gcm_rcm_couple]
    label = gcm_rcm_couple_to_str(gcm_rcm_couple)
    ax.plot(bias_in_the_mean_maxima, altitudes, label=label, color=color)

    return np.array(bias_in_the_mean_maxima)


def main_comparaison_plot():
    altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
    ax = plt.gca()
    gcm_rcm_couples = []

    bias_in_the_mean = []
    for gcm_rcm_couple in gcm_rcm_couples:
        gcm, rcm = gcm_rcm_couple
        years_reanalysis, years_model = get_year_min_and_year_max_used_to_compute_quantile(rcm)
        reanalysis_altitude_studies = AltitudesStudies(study_class=SafranSnowfall2020,
                                                       altitudes=altitudes,
                                                       year_min=years_reanalysis[0],
                                                       year_max=years_reanalysis[1])
        adamont_altitude_studies = AltitudesStudies(study_class=AdamontSnowfall,
                                                    altitudes=altitudes,
                                                    year_min=years_model[0],
                                                    year_max=years_model[1],
                                                    scenario=AdamontScenario.histo,
                                                    gcm_rcm_couple=gcm_rcm_couple)
        bias_in_the_mean.extend(compute_bias_and_display_it(ax, reanalysis_altitude_studies,
                                                            adamont_altitude_studies, gcm_rcm_couple))

    bias_in_the_mean = np.array(bias_in_the_mean)
    min_bias, median_bias, max_bias = [f(bias_in_the_mean, axis=0) for f in [np.min, np.median, np.max]]

    # Plot the range for the bias, and the median
    ax.yaxis.set_ticks(altitudes)
    color = 'k'
    ax.plot(median_bias, altitudes, label='Median bias', color=color)
    ax.fill_betweenx(altitudes, min_bias, max_bias, label='Range for the bias', alpha=0.2, color=color)
    ax.vlines(0, ymin=altitudes[0], ymax=altitudes[-1], color='k')
    study_str = STUDY_CLASS_TO_ABBREVIATION[type(reanalysis_altitude_studies.study)]
    plot_name = 'Bias for {}\n' \
                '(on the period used for the quantile correction)'.format(study_str)
    ax.set_ylim(top=altitudes[-1] + 500)
    ax.legend()
    reanalysis_altitude_studies.show_or_save_to_file(plot_name=plot_name)
    plt.close()


if __name__ == '__main__':
    main_comparaison_plot()
