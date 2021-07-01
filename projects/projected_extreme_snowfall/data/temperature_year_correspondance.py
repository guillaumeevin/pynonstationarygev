import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_averaged_global_mean_temp


def compute_mean_temperature_for_couple_of_period(year_min, year_max):
    d = year_to_averaged_global_mean_temp(AdamontScenario.rcp85_extended, year_min=year_min, year_max=year_max, spline=False, anomaly=True)
    return np.mean(list(d.values()))


def main_mean_temperature():
    for years_couple in [(1986, 2005), (2031, 2050), (2080, 2099)]:
        mean = compute_mean_temperature_for_couple_of_period(*years_couple)
        print(years_couple, mean)

if __name__ == '__main__':
    main_mean_temperature()