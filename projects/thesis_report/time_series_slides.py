from matplotlib import pyplot as plt

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, scenario_to_real_scenarios, \
    get_gcm_rcm_couples
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    ADAMONT_STUDY_CLASS_TO_ABBREVIATION, SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projects.projected_extreme_snowfall.results.setting_utils import load_study_classes
from root_utils import VERSION_TIME


def main_adjustement_coefficients():
    scm_study_class, adamont_study_class = load_study_classes(snowfall=False)
    year_min = 1950
    year_max = 2100
    legend_and_labels = None
    massif_names = ['Vercors']
    season = Season.annual
    adamont_scenario = AdamontScenario.rcp85_extended
    altitudes = [1500]
    for altitude in altitudes:
        plt.figure(figsize=(10, 5))
        # Loading part
        scm_study = scm_study_class(altitude=altitude)
        real_adamont_scenario = scenario_to_real_scenarios(adamont_scenario=adamont_scenario)[-1]

        gcm_rcm_couples = get_gcm_rcm_couples(adamont_scenario=real_adamont_scenario)
        gcm_rcm_couples = gcm_rcm_couples[10:11]
        adamont_studies = AdamontStudies(study_class=adamont_study_class, gcm_rcm_couples=gcm_rcm_couples,
                                         altitude=altitude, year_min_studies=year_min, year_max_studies=year_max,
                                         season=season, scenario=adamont_scenario)
        print(altitude, adamont_scenario)
        adamont_studies.plot_maxima_time_series_adamont(massif_names=massif_names,
                                                        scm_study=scm_study, legend_and_labels=legend_and_labels,
                                                        slides=True)

def main_data_reanalysis():
    for snowfall in [False, True][:]:
        ax = plt.gca()
        study_class, _ = load_study_classes(snowfall=snowfall)
        study = study_class(altitude=1500)
        massif_name = 'Vanoise'
        y = study.massif_name_to_annual_maxima[massif_name]
        x = study.ordered_years
        ax.plot(x, y, color='k', marker='o', linewidth=0, label='S2M reanalysis')

        print(x[-1])

        fontsize = 17
        ax.set_xlabel('Years', fontsize=fontsize)
        abbreviation = 'snowfall' if snowfall else 'snow load'
        plot_name = 'Annual maxima of {}\n in {} at {} m'.format(abbreviation,
                                                               massif_name.replace('_', ' '),
                                                               study.altitude)
        ax.set_ylabel('{} ({})'.format(plot_name, study.variable_unit), fontsize=fontsize)

        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # plt.xticks(rotation=70)
        # ax.tick_params(axis='x', which='major', labelsize=6)
        # ax.set_yticks([0.1 * j for j in range(6)])
        ax.set_xticks(x[::20])
        # ax.set_ylim([0, 220])
        ax.set_xlim([x[0], x[-1]])
        ax.legend()
        filename = '{}/{}/annual maxima'.format(VERSION_TIME, abbreviation)
        StudyVisualizer.savefig_in_results(filename, transparent=False, tight_pad={'h_pad': 0.1})
        plt.close()

def main_data_projections_and_reanalysis():
    for snowfall in [False, True]:
        scm_study_class, adamont_study_class = load_study_classes(snowfall=snowfall)
        year_min = 1950
        year_max = 2100
        legend_and_labels = None
        massif_names = ['Vanoise']
        season = Season.annual
        adamont_scenario = AdamontScenario.rcp85_extended
        altitudes = [1500]
        for altitude in altitudes:
            plt.figure(figsize=(10, 5))
            # Loading part
            scm_study = scm_study_class(altitude=altitude)
            real_adamont_scenario = scenario_to_real_scenarios(adamont_scenario=adamont_scenario)[-1]

            gcm_rcm_couples = get_gcm_rcm_couples(adamont_scenario=real_adamont_scenario)
            adamont_studies = AdamontStudies(study_class=adamont_study_class, gcm_rcm_couples=gcm_rcm_couples,
                                             altitude=altitude, year_min_studies=year_min, year_max_studies=year_max,
                                             season=season, scenario=adamont_scenario)
            print(altitude, adamont_scenario)
            adamont_studies.plot_maxima_time_series_adamont(massif_names=massif_names,
                                                            scm_study=scm_study, legend_and_labels=legend_and_labels,
                                                            slides=True)

def main_data_projections():
    for snowfall in [False, True]:
        scm_study_class, adamont_study_class = load_study_classes(snowfall=snowfall)
        year_min = 1950
        year_max = 2100
        legend_and_labels = None
        massif_names = ['Vanoise']
        season = Season.annual
        adamont_scenario = AdamontScenario.rcp85_extended
        altitudes = [1500]
        for altitude in altitudes:
            plt.figure(figsize=(10, 5))
            # Loading part
            # scm_study = scm_study_class(altitude=altitude)
            real_adamont_scenario = scenario_to_real_scenarios(adamont_scenario=adamont_scenario)[-1]

            gcm_rcm_couples = get_gcm_rcm_couples(adamont_scenario=real_adamont_scenario)
            adamont_studies = AdamontStudies(study_class=adamont_study_class, gcm_rcm_couples=gcm_rcm_couples,
                                             altitude=altitude, year_min_studies=year_min, year_max_studies=year_max,
                                             season=season, scenario=adamont_scenario)
            print(altitude, adamont_scenario)
            adamont_studies.plot_maxima_time_series_adamont(massif_names=massif_names,
                                                            scm_study=None, legend_and_labels=legend_and_labels,
                                                            slides=True)


if __name__ == '__main__':
    # main_data_projections()
    # main_adjustement_coefficients()
    # main_data_reanalysis()
    main_data_projections_and_reanalysis()
