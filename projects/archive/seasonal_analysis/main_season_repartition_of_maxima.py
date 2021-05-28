import matplotlib.pyplot as plt
import operator
import calendar

import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_crocus import AdamontSwe
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import rcm_scenarios_extended, AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSweTotal
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups, \
    get_altitude_group_from_altitudes
import locale


def get_month_name(month_no, locale='C'):
    with calendar.different_locale(locale):
        return calendar.month_name[month_no]


def plot_season_repartition_of_maxima(studies, massif_names, title='', idx=0, projected=False):
    month_to_name = {month: get_month_name(month) for month in range(1, 13)}

    all_years = studies.study.ordered_years
    title += ' on the period '
    if isinstance(studies.study, AbstractAdamontStudy):
        assert studies.study.scenario is AdamontScenario.histo
        title = 'Adamont v2\n' + title
        title += ['HISTORICAL', '', ''][idx]
        years = [all_years, all_years[:30], all_years[-30:]][idx]
    else:
        title = 'SAFRAN 2019\n' + title
        title += ['1959-2019', '-past-1959-1988', '-recent-1990-2019'][idx]
        years = [all_years, all_years[:30], all_years[-30:]][idx]

    color = ['grey', 'red', 'green'][idx]
    ax = plt.gca()
    ax2 = ax.twinx()

    month_to_maxima = get_month_to_maxima(massif_names, studies, years)
    # PLot percentages for each month
    ordered_months = [8, 9, 10, 11, 12] + [1, 2, 3, 4, 5, 6, 7]
    nb_total_maxima = sum([len(v) for v in month_to_maxima.values()])
    percentage_maxima = [100 * len(month_to_maxima[month]) / nb_total_maxima for month in ordered_months]
    assert len(percentage_maxima) == 12
    month_names = [month_to_name[m][:3] for m in ordered_months]
    print(month_names)
    ax.bar(month_names, percentage_maxima, width=0.5,
           color=color, edgecolor=color, label='Percentage of maxima',
           linewidth=2)
    mean_maxima = [np.mean(month_to_maxima[month]) for month in ordered_months]
    # Plot mean maxima for each month
    ax2.plot(month_names, mean_maxima, label='Mean annual maxima')
    ax2.legend(loc='upper right')

    ax.set_ylabel('Percentages of annual maxima')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax2.set_ylabel('Mean annual maxima')
    ax2.set_ylim(bottom=0)

    studies.show_or_save_to_file(title, no_title=True)
    plt.close()


def get_month_to_maxima(massif_names, studies, years):
    month_to_maxima = {i: [] for i in range(1, 13)}
    massif_names = set(massif_names)
    for study in studies.altitude_to_study.values():
        massif_ids = [i for i, m in enumerate(study.study_massif_names) if m in massif_names]
        for year in years:
            annual_maxima = study.year_to_annual_maxima[year][massif_ids]
            annual_maxima_index = study.year_to_annual_maxima_index[year][massif_ids]
            days = study.year_to_days[year]
            days_for_annual_maxima = operator.itemgetter(*annual_maxima_index)(days)
            months = [int(d.split('-')[1]) for d in days_for_annual_maxima]
            for month, annual_maximum in zip(months, annual_maxima):
                month_to_maxima[month].append(annual_maximum)
    return month_to_maxima


def main_repartition_for_snowfall_past():
    study_class = SafranSnowfall1Day
    masssif_names = ['Oisans', 'Grandes-Rousses', 'Haute-Maurienne', 'Vanoise',
                     'Maurienne', 'Belledonne', 'Chartreuse', 'Haute-Tarentaise',
                     'Beaufortain', 'Bauges', 'Mont-Blanc', 'Aravis', 'Chablais',
                     'Mercantour', 'Ubaye', 'Haut_Var-Haut_Verdon', 'Parpaillon', 'Champsaur',
                     'Devoluy', 'Queyras', 'Pelvoux', 'Thabor']
    # 'Vercors'
    norht_massif_names = ['Oisans', 'Grandes-Rousses', 'Haute-Maurienne', 'Vanoise',
                          'Maurienne', 'Belledonne', 'Chartreuse', 'Haute-Tarentaise',
                          'Beaufortain', 'Bauges', 'Mont-Blanc', 'Aravis', 'Chablais']
    south_massif_names = ['Mercantour', 'Ubaye', 'Haut_Var-Haut_Verdon', 'Parpaillon', 'Champsaur',
                          'Devoluy', 'Queyras', 'Pelvoux', 'Thabor']
    for altitudes in altitudes_for_groups[:]:
        studies = AltitudesStudies(study_class, altitudes)
        elevation = get_altitude_group_from_altitudes(altitudes).reference_altitude

        # for idx in range(3):
        # idx enable to plot for some subset of dates
        for idx in range(1):
            for masssif_names, region_name in zip([masssif_names, norht_massif_names, south_massif_names],
                                                  ['All', 'North', 'South']):
                plot_season_repartition_of_maxima(studies, masssif_names, '{} {}'.format(region_name, elevation),
                                                  idx=idx)
                break


def main_repartition_for_swe_projected():
    for study_class in [AdamontSwe, CrocusSweTotal]:
        masssif_names = ['Oisans', 'Grandes-Rousses', 'Haute-Maurienne', 'Vanoise',
                         'Maurienne', 'Belledonne', 'Chartreuse', 'Haute-Tarentaise',
                         'Beaufortain', 'Bauges', 'Mont-Blanc', 'Aravis', 'Chablais',
                         'Mercantour', 'Ubaye', 'Haut_Var-Haut_Verdon', 'Parpaillon', 'Champsaur',
                         'Devoluy', 'Queyras', 'Pelvoux', 'Thabor']
        region_name = 'French Alps'
        for altitudes in altitudes_for_groups[1:]:
            studies = AltitudesStudies(study_class, altitudes)

            if issubclass(study_class, AbstractAdamontStudy):
                print(studies.study.scenario)
            title = 'for elevations ' + get_altitude_group_from_altitudes(altitudes).formula
            # for idx in range(3):
            # idx enable to plot for some subset of dates
            for idx in range(1):
                plot_season_repartition_of_maxima(studies, masssif_names, title,
                                                  idx=idx)


if __name__ == '__main__':
    main_repartition_for_snowfall_past()
    # main_repartition_for_swe_projected()
