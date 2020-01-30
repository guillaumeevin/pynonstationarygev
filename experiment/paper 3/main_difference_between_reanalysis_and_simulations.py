from experiment.meteo_france_data.adamont_data.ensemble_simulation import EnsembleSimulation
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, \
    CrocusSweTotal
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt


def test():
    study = CrocusSnowLoad3Days(altitude=1200)
    study_visualizer = StudyVisualizer(study)
    study_visualizer.visualize_max_graphs_poster('Queyras', altitude='noope', snow_abbreviation="ok", color='red')
    plt.show()


def density_wrt_altitude():
    save_to_file = True
    study_class = CrocusSweTotal
    altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700][::-1]

    ensemble = EnsembleSimulation(first_winter_required_for_histo=1958)

    # None means relative change, true means reanalysis values, false means simulations values
    # mode = False

    for mode in [None, True, False]:
        for altitude in altitudes:

            ax = plt.gca()
            study = study_class(altitude=altitude)
            # study = study_class(altitude=altitude, nb_consecutive_days=3)
            massif_name_to_value = {}
            df = study.observations_annual_maxima.df_maxima_gev.iloc[:2004]
            for massif_name in study.study_massif_names:
                s_reanalysis = df.loc[massif_name].mean()
                if (massif_name, altitude) in ensemble.massif_name_and_altitude_to_mean_average_annual_maxima:
                    s_simulation = ensemble.massif_name_and_altitude_to_mean_average_annual_maxima[(massif_name, altitude)]
                    relative_change_value = 100 * (s_simulation - s_reanalysis) / s_reanalysis
                    if mode == None:
                        value = relative_change_value
                    else:
                        value = s_reanalysis if mode else s_simulation
                    massif_name_to_value[massif_name] = value
            print(massif_name_to_value)
            # Plot
            # massif_name_to_value = {m: i for i, m in enumerate(study.study_massif_names)}
            max_values = max([abs(e) for e in massif_name_to_value.values()]) + 5
            print(max_values)
            variable_name = study.variable_name
            label_relative_change = 'Relative changes in mean annual maxima\n' \
                        'of {}\n between reanalysis and historical simulation\n' \
                        'for 1958-2004  at {}m (%)\n' \
                        ''.format(variable_name, study.altitude)
            if mode is None:
                label = label_relative_change
            else:
                label = 'Mean max SWE for ' + ('reanalysis' if mode else 'simulation')

            vmin = -max_values if mode is None else 0.1
            study.visualize_study(ax=ax, massif_name_to_value=massif_name_to_value,
                                  vmin=vmin, vmax=max_values,
                                  add_colorbar=True,
                                  replace_blue_by_white=False,
                                  show=False,
                                  label=label
                                  )
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file)
            study_visualizer.plot_name = 'relative_changes_in_maxima' if mode is None else label
            study_visualizer.show_or_save_to_file()
            ax.clear()
            plt.close()


if __name__ == '__main__':
    density_wrt_altitude()
    # test()
