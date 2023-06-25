from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepthIn3Days, CrocusSwe3Days
from extreme_data.meteo_france_data.scm_models_data.utils import ORIENTATIONS
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleTemporalModel, NonStationaryLocationTemporalModel, StationaryTemporalModel, \
    NonStationaryScaleTemporalModel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.trend_test.visualizers.utils import load_altitude_to_visualizer

def run_fit(study_class):
    altitudes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300][5:]
    for orientation in ORIENTATIONS[:1]:
        for altitude in altitudes[:1]:
            print('orientation', orientation, 'altitude', altitude)

            studies = AltitudesStudies(study_class, [900], orientation=orientation)
            model_classes = [StationaryTemporalModel, NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel,
                      NonStationaryLocationAndScaleTemporalModel]
            visu = AltitudesStudiesVisualizerForNonStationaryModels(studies,
                                                                    model_classes=model_classes,
                                                                    massif_names=['Mercantour'],
                                                                    remove_physically_implausible_models=True,
                                                                    )

            one_fold_fit = visu._massif_name_to_one_fold_fit['Mercantour']
            print(one_fold_fit.best_estimator)

if __name__ == '__main__':
    for study_class in [CrocusSwe3Days, CrocusDepthIn3Days][1:]:
        run_fit(study_class)
