import matplotlib as mpl
import numpy as np

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days, SafranPrecipitation1Day, SafranPrecipitation3Days, \
    SafranPrecipitation5Days, SafranPrecipitation7Days
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.polynomial_margin_model.utils import ALTITUDINAL_GEV_MODELS, \
    ALTITUDINAL_GUMBEL_MODELS
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels


def plot_altitudinal_fit(studies, massif_names=None):
    model_classes = ALTITUDINAL_GEV_MODELS + ALTITUDINAL_GUMBEL_MODELS
    # model_classes = ALTITUDINAL_GUMBEL_MODELS
    visualizer = AltitudesStudiesVisualizerForNonStationaryModels(studies=studies,
                                                                  model_classes=model_classes,
                                                                  massif_names=massif_names,
                                                                  show=False)
    # Plot the results for the model that minimizes the individual aic
    OneFoldFit.best_estimator_minimizes_mean_aic = False
    visualizer.plot_moments()
    visualizer.plot_shape_map()

    # Compute the mean AIC for each model_class
    OneFoldFit.best_estimator_minimizes_mean_aic = True
    model_class_to_aic_scores = {model_class: [] for model_class in model_classes}
    for one_fold_fit in visualizer.massif_name_to_one_fold_fit.values():
        for model_class, estimator in one_fold_fit.model_class_to_estimator_with_finite_aic.items():
            aic_score_normalized = estimator.aic() / estimator.n()
            model_class_to_aic_scores[model_class].append(aic_score_normalized)
    model_class_to_mean_aic_score = {model_class: np.array(aic_scores).mean()
                                     for model_class, aic_scores in model_class_to_aic_scores.items()}
    print(model_class_to_mean_aic_score)
    sorted_model_class = sorted(model_classes, key=lambda m: model_class_to_mean_aic_score[m])
    best_model_class_for_mean_aic = sorted_model_class[0]
    print(best_model_class_for_mean_aic)
    for one_fold_fit in visualizer.massif_name_to_one_fold_fit.values():
        one_fold_fit.best_estimator_class_for_mean_aic = best_model_class_for_mean_aic

    # Plot the results for the model that minimizes the mean aic
    visualizer.plot_moments()
    visualizer.plot_shape_map()


def plot_time_series(studies, massif_names=None):
    studies.plot_maxima_time_series(massif_names=massif_names)


def plot_moments(studies, massif_names=None):
    for std in [True, False][:]:
        for change in [True, False, None]:
            studies.plot_mean_maxima_against_altitude(massif_names=massif_names, std=std, change=change)


def main():
    # altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][4:7]
    # altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][:]
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900]
    study_classes = [SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, SafranSnowfall7Days][:2]
    study_classes = [SafranPrecipitation1Day, SafranPrecipitation3Days, SafranPrecipitation5Days,
                     SafranPrecipitation7Days][:]
    study_classes = [SafranSnowfall1Day, SafranPrecipitation1Day, SafranSnowfall3Days, SafranPrecipitation3Days][:]
    massif_names = None
    # massif_names = ['Aravis']
    # massif_names = ['Chartreuse', 'Belledonne']

    for study_class in study_classes:
        print('change study class')
        studies = AltitudesStudies(study_class, altitudes, season=Season.winter_extended)
        plot_time_series(studies, massif_names)
        plot_moments(studies, massif_names)
        plot_altitudinal_fit(studies, massif_names)


if __name__ == '__main__':
    main()
