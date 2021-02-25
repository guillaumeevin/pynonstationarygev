from typing import List
import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from extreme_fit.utils import fit_linear_regression
from extreme_trend.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels


def compute_changes_in_total_snowfall(visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels], relative=True):
    changes_in_total_snowfall = []
    for visualizer in visualizer_list:
        changes_for_the_visualizer = []
        for massif_name in visualizer.massif_name_to_one_fold_fit.keys():
            changes_massif = []
            for altitude, study in visualizer.studies.altitude_to_study.items():
                print(altitude)
                if massif_name in study.study_massif_names:
                    change = compute_change_in_total(study, massif_name, relative, plot=False)
                    changes_massif.append(change)
            print(len(changes_massif), 'length')
            mean_change = np.mean(changes_massif)
            changes_for_the_visualizer.append(mean_change)


        changes_in_total_snowfall.append(changes_for_the_visualizer)
    return changes_in_total_snowfall

def compute_change_in_total(study, massif_name, relative, plot=False):
    annual_total = study.massif_name_to_annual_total[massif_name]
    a, b, r2score = fit_linear_regression(study.ordered_years, annual_total)
    years_for_change = [1959, 2019]
    values = [a * y + b for y in years_for_change]
    change = values[1] - values[0]
    if relative:
        change *= 100 / values[0]
    if plot:
        ax = plt.gca()
        ax.plot(study.ordered_years, annual_total)
        linear_plot = [a * y + b for y in study.ordered_years]
        ax.plot(study.ordered_years, linear_plot)
        ax.plot(years_for_change, values, linewidth=0, marker='o')
        ax.set_xlim((study.year_min, study.year_max))
        ax.set_ylabel('Total of snowfall for the {} at {} ({})'.format(massif_name, study.altitude, study.variable_unit))
        plt.show()
        ax.clear()
        plt.close()
    return change


if __name__ == '__main__':
    altitude = 3000
    year_min = 1959
    year_max = 2019
    study = SafranSnowfall(altitude=altitude, year_min=year_min, year_max=year_max)
    print(study.study_massif_names)
    # print(study.massif_name_to_annual_maxima)
    # print(study.year_to_daily_time_serie_array[1959].shape)
    # print(study.massif_name_to_daily_time_series['Vanoise'].shape)
    # study._save_excel_with_longitutde_and_latitude()
    print(study.massif_name_to_annual_total['Vanoise'])
    compute_change_in_total(study, 'Vanoise', relative=True, plot=True)

