import numpy as np
import os.path as op
import matplotlib

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_crocus import AdamontSnowLoad
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import \
    gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, \
    gcm_rcm_couple_to_str, get_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_max_swe import CrocusSnowLoad2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2019
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies


def compute_bias_and_display_it(ax,
                                altitude_studies_reanalysis: AltitudesStudies,
                                adamont_altitude_studies: AltitudesStudies,
                                gcm_rcm_couple,
                                massif_names=None,
                                            relative_bias=False,
                                mean=True,
                                ):
    bias_in_the_mean_maxima = []
    altitudes = []
    for altitude, study_reanalysis in altitude_studies_reanalysis.altitude_to_study.items():
        adamont_study = adamont_altitude_studies.altitude_to_study[altitude]
        can_compute_biais = (massif_names is None) or any([m in adamont_study.study_massif_names for m in massif_names])
        if can_compute_biais:

            altitudes.append(altitude)

            f = np.mean if mean else np.std
            if massif_names is None:
                mean_maxima_adamont = adamont_study.aggregate_annual_maxima(f)
                mean_maxima_reanalysis = study_reanalysis.aggregate_annual_maxima(f)
            else:
                mean_maxima_reanalysis = f(np.concatenate([study_reanalysis.massif_name_to_annual_maxima[m]
                                                           for m in massif_names
                                                           if m in study_reanalysis.massif_name_to_annual_maxima]))
                mean_maxima_adamont = f(np.concatenate([adamont_study.massif_name_to_annual_maxima[m]
                                                        for m in massif_names
                                                        if m in adamont_study.massif_name_to_annual_maxima]))

            bias = mean_maxima_adamont - mean_maxima_reanalysis
            if relative_bias:
                bias *= 100 / mean_maxima_reanalysis
            bias_in_the_mean_maxima.append(bias)

    color = gcm_rcm_couple_to_color[gcm_rcm_couple]
    label = gcm_rcm_couple_to_str(gcm_rcm_couple)
    ax.plot(bias_in_the_mean_maxima, altitudes, label=label, color=color)

    return np.array(bias_in_the_mean_maxima), altitudes


def main_comparaison_plot():

    for adamont_version in [2][:]:
        print('version:', adamont_version)


        for gcm_as_pseudo_truth in [None, ("MPI-ESM-LR", "CCLM4-8-17"), ("CNRM-CM5", "ALADIN53")]:
            gcm_rcm_couples = get_gcm_rcm_couples(adamont_scenario=AdamontScenario.histo,
                                                  adamont_version=adamont_version)

            if gcm_as_pseudo_truth is not None:
                gcm_rcm_couples.remove(gcm_as_pseudo_truth)

            snowfall = False
            if snowfall:
                safran_study_class = SafranSnowfall2019 if adamont_version == 2 else SafranSnowfall1Day
                adamont_study_class = AdamontSnowfall
                altitudes = [1200, 2100, 3000][:]
            else:
                safran_study_class = CrocusSnowLoad2019
                adamont_study_class = AdamontSnowLoad
                altitudes = [600, 900, 1200, 1500, 1800][-2:]

            assert (safran_study_class is None) or (issubclass(safran_study_class, AbstractStudyMaxFiles))

            comparaison_study_class = 'SAFRAN'

            year_min_to_reanalysis_altitude_studies = {}

            # list_of_massis_names = [None]
            list_of_massis_names = [['Beaufortain']]
            list_of_massis_names = [['Haute-Tarentaise']]
            list_of_massis_names = [['Vanoise']]

            for relative_bias in [True, False][:]:
                for mean in [True, False][:]:
                    for massif_names in list_of_massis_names:


                        ax = plt.gca()
                        bias_in_the_mean = []
                        list_altitudes_for_bias = []
                        for gcm_rcm_couple in gcm_rcm_couples:
                            adamont_altitude_studies = AltitudesStudies(study_class=adamont_study_class,
                                                                        altitudes=altitudes,
                                                                        year_min=1959,
                                                                        year_max=2019,
                                                                        scenario=AdamontScenario.histo,
                                                                        gcm_rcm_couple=gcm_rcm_couple,
                                                                        adamont_version=adamont_version)
                            year_min = adamont_altitude_studies.study.year_min
                            if year_min not in year_min_to_reanalysis_altitude_studies:
                                if gcm_as_pseudo_truth is None:
                                    studies = AltitudesStudies(study_class=safran_study_class,
                                                     altitudes=altitudes,
                                                     year_min=year_min,
                                                     year_max=2019)
                                else:
                                    studies = AltitudesStudies(study_class=adamont_study_class,
                                                                        altitudes=altitudes,
                                                                        year_min=year_min,
                                                                        year_max=2019,
                                                                        scenario=AdamontScenario.histo,
                                                                        gcm_rcm_couple=gcm_as_pseudo_truth,
                                                                        adamont_version=adamont_version)
                                year_min_to_reanalysis_altitude_studies[year_min] = studies

                            bias, altitudes_for_bias = compute_bias_and_display_it(ax, year_min_to_reanalysis_altitude_studies[year_min],
                                                               adamont_altitude_studies, gcm_rcm_couple, massif_names,
                                                                                   relative_bias, mean)
                            bias_in_the_mean.append(bias)
                            list_altitudes_for_bias.append(altitudes_for_bias)

                        # Assert the all the bias have been computed for the same altitudes
                        altitudes_for_bias = list_altitudes_for_bias[0]
                        for alti in list_altitudes_for_bias:
                            assert alti == altitudes_for_bias

                        bias_in_the_mean = np.array(bias_in_the_mean)
                        min_bias, median_bias, mean_bias, max_bias = [f(bias_in_the_mean, axis=0) for f in [np.min, np.median, np.mean, np.max]]

                        reanalysis_altitude_studies = year_min_to_reanalysis_altitude_studies[1959]
                        # Plot the range for the bias, and the median
                        ax.yaxis.set_ticks(altitudes)
                        color = 'k'
                        # ax.plot(median_bias, altitudes_for_bias, label='Median bias', color=color, linewidth=4)
                        ax.plot(mean_bias, altitudes_for_bias, label='Average bias', color=color, linewidth=4)
                        # ax.fill_betweenx(altitudes, min_bias, max_bias, label='Range for the bias', alpha=0.2, color='whitesmoke')
                        ax.vlines(0, ymin=altitudes[0], ymax=altitudes[-1], color='k', linestyles='dashed')
                        study_str = STUDY_CLASS_TO_ABBREVIATION[type(reanalysis_altitude_studies.study)]
                        ax.set_ylim(top=altitudes[-1] + 800)
                        study = adamont_altitude_studies.study
                        ax.legend(ncol=3, prop={'size': 7})
                        ax.set_ylabel('Altitude (m)', fontsize=10)
                        massif_str = 'all massifs' if massif_names is None else 'the {} massif '.format(massif_names[0])
                        unit = '%' if relative_bias else study.variable_unit
                        # bias_name = 'Relative differences' if relative_bias else 'Differences'
                        bias_name = 'Relative biases' if relative_bias else 'Biases'
                        if gcm_as_pseudo_truth is None:
                            bias_name += ' w.r.t. the observations'
                        else:
                            bias_name += ' w.r.t. the pseudo observations from {}'.format('/'.join(gcm_as_pseudo_truth))
                        mean_str = 'mean' if mean else 'std'
                        title = '{} in the {} annual\nmaxima of {} for {}' \
                                'on the period 1959-2019 ({})'.format(bias_name, mean_str,
                                                                                                  study_str, massif_str,
                                                                                                  unit)
                        folder = 'relative difference' if relative_bias else 'difference'
                        plot_name = op.join(folder, title)
                        ax.set_xlabel(title, fontsize=10)
                        reanalysis_altitude_studies.show_or_save_to_file(plot_name=plot_name, no_title=True)
                        plt.close()


if __name__ == '__main__':
    main_comparaison_plot()
