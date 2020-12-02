import numpy as np
import matplotlib.pyplot as plt
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranTemperature, \
    SafranPrecipitation1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.snowfall_plot import \
    fit_linear_regression


def delta_graph(study_class, altitudes, maxima=True):
    ax = plt.gca()
    all_delta_point_for_regression = []
    # colors = ['orange', 'red', 'blue', 'green', 'yellow']
    for altitude in altitudes:
        delta_points = []
        study = study_class(altitude=altitude)  # type : AbstractStudy
        study_temperature = SafranTemperature(altitude=altitude)
        for massif_name in study.study_massif_names:
            if maxima:
                values = study.massif_name_to_annual_maxima[massif_name]
            else:
                values = study.massif_name_to_annual_total[massif_name]
            values_temperature = study_temperature.massif_name_to_annual_total[massif_name]
            delta_point = (compute_delta(values_temperature, relative=False), compute_delta(values, relative=True))
            delta_points.append(delta_point)
        all_delta_point_for_regression.extend(delta_points)
        # Plot part
        x, y = list(zip(*delta_points))
        ax.scatter(x, y, label='{}m'.format(altitude))
        aggregation_str = 'maxima' if maxima else 'total'
        ylabel = 'Delta for {} {}'.format(aggregation_str, STUDY_CLASS_TO_ABBREVIATION[study_class])
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Delta mean Temperature')
    # Plot linear regression
    x_all, y_all = list(zip(*all_delta_point_for_regression))
    a, b, r2_score = fit_linear_regression(x_all, y_all)
    a = a[0]
    x_plot = np.linspace(start=np.min(x_all), stop=np.max(x_all), num=100)
    y_plot = a * x_plot + b
    rounded_number = [str(np.round(e, 2)) for e in [a, b, r2_score]]
    ax.plot(x_plot, y_plot, label='{} x + {} (R2 = {})'.format(*rounded_number))
    visualizer = StudyVisualizer(study=study, show=False, save_to_file=True)
    visualizer.plot_name = ylabel
    ax.legend()

    # Show / Save plot
    visualizer.show_or_save_to_file(no_title=True)
    plt.close()


def compute_delta(values, relative=True):
    index = 30
    before, after = values[:index], values[index:]
    mean_before, mean_after = np.mean(before), np.mean(after)
    delta = mean_after - mean_before
    if relative:
        delta *= 100 / mean_before
    return delta


if __name__ == '__main__':
    fast = False
    if fast is None:
        altitudes = [900, 1800, 2700]
    elif fast:
        altitudes = [900]
    else:
        altitudes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600][:]
    for study_class in [SafranSnowfall1Day, SafranPrecipitation1Day]:
        for maxima in [True, False]:
            delta_graph(study_class, altitudes, maxima=maxima)
