import matplotlib.pyplot as plt
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.safran.gap_between_study import \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfallCenterOnDay, \
    SafranSnowfallCenterOnDay1day
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    STUDY_CLASS_TO_ABBREVIATION


def plot_comparison(altitudes, massif_name, stud_class_list):
    altitude_studies_list = [AltitudesStudies(study_class, altitudes) for study_class in stud_class_list]

    ax = plt.gca()

    for altitude_studies in altitude_studies_list:
        plot_studies(ax, altitude_studies, massif_name)

    labelsize = 10

    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], prop={'size': labelsize})

    plot_name = 'comparaison for {} at {}'.format(massif_name, altitudes[0])
    altitude_studies_list[0].show_or_save_to_file(plot_name=plot_name, show=False, no_title=True, tight_layout=True)
    ax.clear()
    plt.close()

def plot_studies(ax, altitude_studies, massif_name):
    for altitude, study in list(altitude_studies.altitude_to_study.items()):
        x = study.ordered_years
        if massif_name in study.massif_name_to_annual_maxima:
            y = study.massif_name_to_annual_maxima[massif_name]
            label = '{} m'.format(altitude)
            ax.plot(x, y, linewidth=2, label=label)
        study_class = type(study)
        plot_name = 'Annual maxima of {} in {}'.format(STUDY_CLASS_TO_ABBREVIATION[study_class],
                                                       massif_name.replace('_', ' '))
        fontsize = 10
        ax.set_ylabel('{} ({})'.format(plot_name, study.variable_unit), fontsize=fontsize)


if __name__ == '__main__':
    for study_class in [GapBetweenSafranSnowfall2019AndMySafranSnowfall2019, SafranSnowfall1Day, SafranSnowfallCenterOnDay1day][:1]:
        for altitudes, massif_name in [([2100, 2400, 2700], 'Chablais'), ([1200, 1500, 1800], 'Vercors')][:]:
            plot_comparison(altitudes, massif_name, [study_class])