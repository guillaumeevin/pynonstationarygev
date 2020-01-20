from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth, CrocusSnowLoad3Days, \
    CrocusSnowLoadTotal
from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusDepthVariable
from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, SafranRainfall, SafranTotalPrecip
from experiment.meteo_france_data.scm_models_data.safran.safran_variable import SafranTotalPrecipVariable
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDY_CLASS_TO_ABBREVIATION, snow_density_str, ALL_ALTITUDES_WITHOUT_NAN
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt

from experiment.paper_past_snow_loads.discussion_data_comparison_with_eurocode.crocus_study_comparison_with_eurocode import \
    CrocusDifferenceSnowLoad, \
    CrocusSnowDensityAtMaxofSwe, CrocusDifferenceSnowLoadRescaledAndEurocodeToSeeSynchronization, \
    CrocusSnowDepthAtMaxofSwe, CrocusSnowDepthDifference
from experiment.paper_past_snow_loads.paper_utils import dpi_paper1_figure


def density_wrt_altitude():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = False
    study_class = [SafranTotalPrecip, SafranRainfall, SafranSnowfall, CrocusSnowLoad3Days, CrocusSnowLoadTotal][0]
    # altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][::-1]
    altitudes = ALL_ALTITUDES_WITHOUT_NAN[1:]
    altitudes = [900]



    for altitude in altitudes:

        ax = plt.gca()
        study = study_class(altitude=altitude, nb_consecutive_days=3)
        massif_name_to_value = {}
        for massif_name in study.study_massif_names:
            s = study.observations_summer_annual_maxima.df_maxima_gev.loc[massif_name]
            year_limit = 1987
            print(s)
            df_before, df_after = s.loc[:year_limit], s.loc[year_limit+1:]
            print(df_before, df_after)

            df_before, df_after = df_before.median(), df_after.median()
            relative_change_value = 100 * (df_after - df_before) / df_before
            massif_name_to_value[massif_name] = relative_change_value
        print(massif_name_to_value)

        # Plot
        # massif_name_to_value = {m: i for i, m in enumerate(study.study_massif_names)}
        max_values = max([abs(e) for e in massif_name_to_value.values()]) + 5
        print(max_values)
        study.visualize_study(ax=ax, massif_name_to_value=massif_name_to_value,
                              vmin=-max_values, vmax=max_values,
                              add_colorbar=True,
                              replace_blue_by_white=False,
                              label='Relative changes for \n{}\n at {}m (%)'.format(study.variable_name, study.altitude)
                              )
        plt.show()
        ax.clear()


if __name__ == '__main__':
    density_wrt_altitude()
