from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, \
    CrocusSnowLoadTotal
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, SafranRainfall, SafranPrecipitation
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt


def test():
    study = CrocusSnowLoad3Days(altitude=1200)
    study_visualizer = StudyVisualizer(study)
    study_visualizer.visualize_max_graphs_poster('Queyras', altitude='noope', snow_abbreviation="ok", color='red')
    plt.show()


def density_wrt_altitude():
    save_to_file = True
    study_class = [SafranPrecipitation, SafranRainfall, SafranSnowfall, CrocusSnowLoad3Days, CrocusSnowLoadTotal][-2]
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][::-1]

    for altitude in altitudes:

        ax = plt.gca()
        study = study_class(altitude=altitude)
        # study = study_class(altitude=altitude, nb_consecutive_days=3)
        massif_name_to_value = {}
        for massif_name in study.study_massif_names:
            s = study.observations_annual_maxima.df_maxima_gev.loc[massif_name]
            year_limit = 1987
            df_before, df_after = s.loc[:year_limit], s.loc[year_limit + 1:]
            df_before, df_after = df_before.mean(), df_after.mean()
            # df_before, df_after = df_before.median(), df_after.median()
            relative_change_value = 100 * (df_after - df_before) / df_before
            massif_name_to_value[massif_name] = relative_change_value
        print(massif_name_to_value)
        # Plot
        # massif_name_to_value = {m: i for i, m in enumerate(study.study_massif_names)}
        max_values = max([abs(e) for e in massif_name_to_value.values()]) + 5
        print(max_values)
        variable_name = study.variable_name
        study.visualize_study(ax=ax, massif_name_to_value=massif_name_to_value,
                              vmin=-max_values, vmax=max_values,
                              add_colorbar=True,
                              replace_blue_by_white=False,
                              show=False,
                              label='Relative changes in mean annual maxima\n'
                                    'of {}\n between 1958-1987 and 1988-2017\n  at {}m (%)\n'
                                    ''.format(variable_name, study.altitude)
                              )
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file)
        study_visualizer.plot_name = 'relative_changes_in_maxima'
        study_visualizer.show_or_save_to_file()
        ax.clear()
        plt.close()


if __name__ == '__main__':
    density_wrt_altitude()
    # test()
