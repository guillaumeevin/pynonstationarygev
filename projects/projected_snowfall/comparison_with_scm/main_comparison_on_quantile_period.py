import numpy as np
import matplotlib

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import \
    get_year_min_and_year_max_used_to_compute_quantile
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, gcm_rcm_couple_to_color, \
    gcm_rcm_couple_to_str, load_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020

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
    altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600][:]
    for adamont_version in [1, 2]:
        ax = plt.gca()
        gcm_rcm_couples = load_gcm_rcm_couples(adamont_scenario=AdamontScenario.histo, adamont_version=adamont_version)

        study_class = SafranSnowfall2020 if adamont_version == 2 else SafranSnowfall1Day
        comparaison_study_class = 'SAFRAN 2020' if adamont_version == 2 else 'SAFRAN 2019'

        # Faster to load once the two cases
        reanalysis_altitude_studies_1981 = AltitudesStudies(study_class=study_class,
                                                            altitudes=altitudes,
                                                            year_min=1981,
                                                            year_max=2011)
        reanalysis_altitude_studies_1988 = AltitudesStudies(study_class=study_class,
                                                            altitudes=altitudes,
                                                            year_min=1988,
                                                            year_max=2011)
        bias_in_the_mean = []
        for gcm_rcm_couple in gcm_rcm_couples:
            print(gcm_rcm_couple)
            gcm, rcm = gcm_rcm_couple
            years_reanalysis, years_model = get_year_min_and_year_max_used_to_compute_quantile(gcm)
            assert years_reanalysis[0] in [1981, 1988]
            if years_reanalysis[0] == 1981:
                reanalysis_altitude_studies = reanalysis_altitude_studies_1981
            else:
                reanalysis_altitude_studies = reanalysis_altitude_studies_1988
            adamont_altitude_studies = AltitudesStudies(study_class=AdamontSnowfall,
                                                        altitudes=altitudes,
                                                        year_min=years_model[0],
                                                        year_max=years_model[1],
                                                        scenario=AdamontScenario.histo,
                                                        gcm_rcm_couple=gcm_rcm_couple,
                                                        adamont_version=adamont_version)
            bias_in_the_mean.append(compute_bias_and_display_it(ax, reanalysis_altitude_studies,
                                                                adamont_altitude_studies, gcm_rcm_couple))

        bias_in_the_mean = np.array(bias_in_the_mean)
        min_bias, median_bias, max_bias = [f(bias_in_the_mean, axis=0) for f in [np.min, np.median, np.max]]

        # Plot the range for the bias, and the median
        ax.yaxis.set_ticks(altitudes)
        color = 'k'
        ax.plot(median_bias, altitudes, label='Median bias', color=color, linewidth=4)
        # ax.fill_betweenx(altitudes, min_bias, max_bias, label='Range for the bias', alpha=0.2, color='whitesmoke')
        ax.vlines(0, ymin=altitudes[0], ymax=altitudes[-1], color='k', linestyles='dashed')
        study_str = STUDY_CLASS_TO_ABBREVIATION[type(reanalysis_altitude_studies.study)]
        plot_name = 'Bias for annual maxima of {}'.format(study_str)
        ax.set_ylim(top=altitudes[-1] + 1300)
        study = adamont_altitude_studies.study
        ax.legend(ncol=3, prop={'size': 7})
        ax.set_ylabel('Altitude (m)', fontsize=10)
        ax.set_xlabel('Bias in the mean annual maxima of {} for ADAMONT v{} members\n'
                      ' against {} on the quantile mapping period ({})'.format(study_str, adamont_version,
                                                                               comparaison_study_class,
                                                                               study.variable_unit), fontsize=10)
        reanalysis_altitude_studies.show_or_save_to_file(plot_name=plot_name, no_title=True)
        plt.close()


if __name__ == '__main__':
    main_comparaison_plot()
