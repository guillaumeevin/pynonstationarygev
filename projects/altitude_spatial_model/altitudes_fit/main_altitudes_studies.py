from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days, SafranPrecipitation1Day, SafranPrecipitation3Days, \
    SafranPrecipitation5Days, SafranPrecipitation7Days
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.polynomial_margin_model.utils import ALTITUDINAL_MODELS
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels


def plot_altitudinal_fit(studies):
    visualizer = AltitudesStudiesVisualizerForNonStationaryModels(studies=studies,
                                                                  model_classes=ALTITUDINAL_MODELS,
                                                                  massif_names=['Belledonne'],
                                                                  show=True)
    visualizer.plot_mean()
    visualizer.plot_relative_change()


def plot_time_series(studies):
    studies.plot_maxima_time_series()


def plot_moments(studies):
    for std in [True, False][1:]:
        for change in [True, False, None]:
            studies.plot_mean_maxima_against_altitude(std=std, change=change)


def main():
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    # altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900]
    study_classes = [SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, SafranSnowfall7Days][:2]
    study_classes = [SafranPrecipitation1Day, SafranPrecipitation3Days, SafranPrecipitation5Days,
                     SafranPrecipitation7Days][:]
    study_classes = [SafranPrecipitation1Day, SafranSnowfall1Day, SafranSnowfall3Days, SafranPrecipitation3Days][:1]

    for study_class in study_classes:
        studies = AltitudesStudies(study_class, altitudes, season=Season.winter_extended)
        # plot_time_series(studies)
        # plot_moments(studies)
        plot_altitudinal_fit(studies)


if __name__ == '__main__':
    main()
