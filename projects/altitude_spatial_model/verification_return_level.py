import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day


def compute_return_level():
    count_all = 0
    count_exceedance = 0
    diff_list = []
    for altitude in [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600][:]:
        study = SafranSnowfall1Day(altitude=altitude, year_max=2008)
        for massif_name in study.study_massif_names:
            if massif_name in study.massif_name_to_stationary_gev_params:

                gev_params = study.massif_name_to_stationary_gev_params[massif_name]
                return_level100 = gev_params.return_level(return_period=100)
                annual_maxima = study.massif_name_to_annual_maxima[massif_name]
                annual_maxima = sorted(annual_maxima)
                max_annual_maxima = max(annual_maxima)

                count_all += 1
                relative_diff =100 *  (return_level100 - max_annual_maxima) / max_annual_maxima
                diff_list.append(relative_diff)

                if return_level100 < max_annual_maxima:
                    count_exceedance += 1
                    print(altitude, massif_name, annual_maxima[-2:], return_level100, gev_params.shape)
    percent_of_exceedance = 100 * count_exceedance / count_all
    print(percent_of_exceedance)
    mean_relatif_diff = np.mean(diff_list)
    print(mean_relatif_diff)

if __name__ == '__main__':
    compute_return_level()