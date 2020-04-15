import pandas as pd
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, \
    CrocusSnowLoadTotal
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, SafranRainfall, \
    SafranPrecipitation, SafranPrecipitation1Day, SafranSnowfall1Day, SafranTemperature, \
    SafranNormalizedPreciptationRateOnWetDays, SafranNormalizedPreciptationRate
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import \
    SafranNormalizedPrecipitationRateOnWetDaysVariable
from extreme_data.meteo_france_data.scm_models_data.utils import SeasonForTheMaxima
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt


def relative_change_in_maxima_wrt_altitude():
    save_to_file = True
    study_class = [SafranSnowfall1Day, SafranPrecipitation1Day, SafranPrecipitation, SafranRainfall,
                   SafranSnowfall, CrocusSnowLoad3Days, CrocusSnowLoadTotal,
                   SafranTemperature, SafranNormalizedPreciptationRateOnWetDays,
                   SafranNormalizedPreciptationRate][1]
    altitudes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][::-1]
    relative = True

    for altitude in altitudes:

        ax = plt.gca()
        study = study_class(altitude=altitude, season=SeasonForTheMaxima.winter_extended)
        # study = study_class(altitude=altitude, nb_consecutive_days=3)
        massif_name_to_value = {}
        for massif_name in study.study_massif_names:
            if study_class is SafranTemperature:
                s = study.observations_annual_mean.loc[massif_name]
            else:
                s = study.observations_annual_maxima.df_maxima_gev.loc[massif_name]
            year_limit = 1989
            df_before, df_after = s.loc[:year_limit], s.loc[year_limit + 1:]
            df_before, df_after = df_before.mean(), df_after.mean()
            # df_before, df_after = df_before.median(), df_after.median()
            if relative:
                change_value = 100 * (df_after - df_before) / df_before
            else:
                change_value = (df_after - df_before)
            massif_name_to_value[massif_name] = change_value
        print(massif_name_to_value)
        # Plot
        # massif_name_to_value = {m: i for i, m in enumerate(study.study_massif_names)}
        max_values = max([abs(e) for e in massif_name_to_value.values()]) * 1.05
        print(max_values)
        variable_name = study.variable_name
        prefix = 'Relative' if relative else ''
        str_season = str(study.season).split('.')[-1]
        study.visualize_study(ax=ax, massif_name_to_value=massif_name_to_value,
                              vmin=-max_values, vmax=max_values,
                              add_colorbar=True,
                              replace_blue_by_white=False,
                              show=False,
                              label='{} changes in mean {} maxima\n'
                                    'of {}\n between 1959-1989 and 1990-2019\n  at {}m (%)\n'
                                    ''.format(prefix, str_season, variable_name, study.altitude)
                              )
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file)
        study_visualizer.plot_name = '{}_changes_in_maxima'.format(prefix)
        study_visualizer.show_or_save_to_file()
        ax.clear()
        plt.close()


if __name__ == '__main__':
    relative_change_in_maxima_wrt_altitude()
    # test()
