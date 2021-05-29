import matplotlib as mpl

from projects.projected_extreme_snowfall.results.setting_utils import load_study_classes

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples, \
    scenario_to_real_scenarios
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    scm_study_class, adamont_study_class = load_study_classes(snowfall=False)
    year_min = 1950
    year_max = 2100
    legend_and_labels = True
    massif_names = ['Vanoise']
    season = Season.annual
    adamont_scenario = AdamontScenario.rcp85_extended
    altitudes = [600, 1200, 1800]
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
                                                        scm_study=scm_study, legend_and_labels=legend_and_labels)

if __name__ == '__main__':
    main()
