from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_full_name, \
    get_year_min_and_year_max_from_scenario, AdamontScenario
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from projects.projected_snowfall.comparison_with_scm.comparison_historical_visualizer import \
    ComparisonHistoricalVisualizer


def load_historical_adamont_studies(study_class, year_min, year_max):
    gcm_rcm_couples = []
    for gcm_rcm_couple in gcm_rcm_couple_to_full_name.keys():
        year_min_couple, year_max_couple = get_year_min_and_year_max_from_scenario(adamont_scenario=AdamontScenario.histo,
                                                                                   gcm_rcm_couple=gcm_rcm_couple)
        if year_min_couple <= year_min and year_max <= year_max_couple:
            gcm_rcm_couples.append(gcm_rcm_couple)
    return AdamontStudies(study_class, gcm_rcm_couples, year_min=year_min, year_max=year_max)


def main():
    fast = [True, False][0]
    # Set the year_min and year_max for the comparison
    if fast:
        year_min = [1982, 1950][1]
        massif_names = ['Vanoise']
        altitudes = [1800]
    else:
        massif_names = None
        year_min = [1982, 1950][1]
        altitudes = [1800, 2100]

    for altitude in altitudes:
        plot(altitude, massif_names, year_min)


def plot(altitude, massif_names, year_min):
    year_min = max(1959, year_min)
    year_max = 2005
    study_class_couple = [(SafranSnowfall1Day, AdamontSnowfall)][0]
    scm_study_class, adamont_study_class = study_class_couple
    scm_study = scm_study_class(altitude=altitude, year_min=year_min, year_max=year_max)
    adamont_studies = load_historical_adamont_studies(adamont_study_class, year_min, year_max)

    visualizer = ComparisonHistoricalVisualizer(scm_study, adamont_studies, massif_names=massif_names)
    for plot_maxima in [True, False][1:]:
        visualizer.plot_comparison(plot_maxima)


if __name__ == '__main__':
    main()