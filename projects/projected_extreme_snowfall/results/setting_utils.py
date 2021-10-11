from typing import List

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_crocus import AdamontSnowLoad
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall, AdamontPrecipitation
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_max_swe import CrocusSnowLoad2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_precipf import SafranPrecipitation2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2019
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel, NonStationaryShapeTemporalModel, \
    NonStationaryScaleAndShapeTemporalModel, NonStationaryLocationAndScaleAndShapeTemporalModel, \
    NonStationaryLocationAndShapeTemporalModel, NonStationaryLocationAndScaleTemporalModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationModel, NonStationaryTwoLinearScaleOneLinearShapeModel, \
    NonStationaryTwoLinearScaleAndShapeModel, NonStationaryTwoLinearShapeOneLinearLocAndScaleModel, \
    NonStationaryTwoLinearScaleOneLinearLocAndShapeModel, NonStationaryTwoLinearShapeOneLinearLocModel, \
    NonStationaryTwoLinearScaleOneLinearLocModel, NonStationaryTwoLinearScaleAndShapeOneLinearLocModel, \
    NonStationaryTwoLinearLocationOneLinearScaleModel, NonStationaryTwoLinearLocationOneLinearScaleAndShapeModel, \
    NonStationaryTwoLinearLocationOneLinearShapeModel, NonStationaryTwoLinearLocationAndShapeOneLinearScaleModel, \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel, \
    NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel, NonStationaryTwoLinearLocationAndScaleModel, \
    NonStationaryTwoLinearLocationAndShape, NonStationaryThreeLinearLocationAndScaleAndShapeModel, \
    NonStationaryFourLinearLocationAndScaleAndShapeModel, NonStationaryFiveLinearLocationAndScaleAndShapeModel, \
    NonStationarySixLinearLocationAndScaleAndShapeModel, NonStationarySevenLinearLocationAndScaleAndShapeModel, \
    NonStationaryTenLinearLocationAndScaleAndShapeModel, NonStationaryEightLinearLocationAndScaleAndShapeModel, \
    NonStationaryNineLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearShapeModel, NonStationaryTwoLinearShapeOneLinearScaleModel, NonStationaryTwoLinearScaleModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_fit.model.utils import set_seed_for_test
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from root_utils import get_display_name_from_object_type
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


def set_up_and_load(fast, snowfall=True):
    # todo: it might have been prerable to load an object containing all the attributes
    safran_study_class, study_class = load_study_classes(snowfall)
    OneFoldFit.multiprocessing = False

    remove_physically_implausible_models, display_only_model_that_pass_gof_test = False, False

    fit_method = MarginFitMethod.evgam

    model_classes = [StationaryTemporalModel]
    model_classes = [NonStationaryLocationAndScaleAndShapeTemporalModel]
    # model_classes = [NonStationaryTwoLinearLocationAndScaleAndShapeModel]
    # model_classes = [NonStationaryThreeLinearLocationAndScaleAndShapeModel]
    # model_classes = [NonStationaryFourLinearLocationAndScaleAndShapeModel]

    altitudes_list = [1500]

    if snowfall is True:
        return_period = 100
    elif snowfall is None:
        return_period = 100
    else:
        return_period = 50
    OneFoldFit.return_period = return_period

    temporal_covariate_for_fit = [TimeTemporalCovariate,
                                  AnomalyTemperatureWithSplineTemporalCovariate][1]
    set_seed_for_test()
    scenario = AdamontScenario.rcp85_extended
    AltitudesStudiesVisualizerForNonStationaryModels.consider_at_least_two_altitudes = False
    gcm_rcm_couples = get_gcm_rcm_couples(scenario)
    print('Scenario is', scenario)
    print('Covariate is {}'.format(temporal_covariate_for_fit))
    if fast is None:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = ["Vanoise"]
        gcm_rcm_couples = gcm_rcm_couples[:]
    elif fast:
        gcm_rcm_couples = gcm_rcm_couples[:3] + gcm_rcm_couples[-3:]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = ['Vanoise']
    else:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 100
        massif_names = AbstractStudy.all_massif_names()

    assert isinstance(gcm_rcm_couples, list)
    altitudes_list = [[a] for a in altitudes_list]
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    assert (safran_study_class is None) or (issubclass(safran_study_class, AbstractStudyMaxFiles))

    print('Altitudes', altitudes_list)
    print('Return period:', OneFoldFit.return_period)
    print('Significance level:', OneFoldFit.SIGNIFICANCE_LEVEL)
    for altitudes in altitudes_list:
        assert len(altitudes) == 1
    print('number of models', len(model_classes))
    print('first model', get_display_name_from_object_type(model_classes[0]))
    print('number of gcm rcm couples', len(gcm_rcm_couples))

    print('only models that pass gof:', display_only_model_that_pass_gof_test)
    print('remove physically implausible models:', remove_physically_implausible_models)

    print('observation class:', get_display_name_from_object_type(safran_study_class))
    print('Take into account the observations: {}'.format(safran_study_class is not None))

    return altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, study_class, temporal_covariate_for_fit, remove_physically_implausible_models, display_only_model_that_pass_gof_test, safran_study_class, fit_method


def load_study_classes(snowfall):

    if snowfall is True:
        safran_study_class = [None, SafranSnowfall2019][1]  # None means we do not account for the observations
        study_class = AdamontSnowfall
    elif snowfall is None:
        study_class = AdamontPrecipitation
        safran_study_class = SafranPrecipitation2019
    else:
        study_class = AdamontSnowLoad
        safran_study_class = CrocusSnowLoad2019
    return safran_study_class, study_class


def get_last_year_for_the_train_set(percentage):
    last_year_for_the_train_set = 1959 + round(percentage * 61) - 1
    return last_year_for_the_train_set


LINEAR_MODELS_FOR_PROJECTION_ONE_ALTITUDE= [
    StationaryTemporalModel,
    NonStationaryScaleTemporalModel,
    NonStationaryShapeTemporalModel,
    NonStationaryLocationTemporalModel,
    NonStationaryLocationAndScaleTemporalModel,
    NonStationaryLocationAndShapeTemporalModel,
    NonStationaryScaleAndShapeTemporalModel,
    NonStationaryLocationAndScaleAndShapeTemporalModel,
]

SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE = [

    # Models with a constant Location parameter
    StationaryTemporalModel,
    # Simple linearity for the others
    NonStationaryScaleTemporalModel,
    NonStationaryShapeTemporalModel,
    NonStationaryScaleAndShapeTemporalModel,
    # Double linearity for the others
    NonStationaryTwoLinearScaleModel,
    NonStationaryTwoLinearShapeModel,
    NonStationaryTwoLinearShapeOneLinearScaleModel,
    NonStationaryTwoLinearScaleOneLinearShapeModel,
    NonStationaryTwoLinearScaleAndShapeModel,

    # Models with a linear location parameter
    NonStationaryLocationTemporalModel,
    # Simple linearity for the others
    NonStationaryLocationAndScaleTemporalModel,
    NonStationaryLocationAndShapeTemporalModel,
    NonStationaryLocationAndScaleAndShapeTemporalModel,
    # Double linearity for the others
    NonStationaryTwoLinearScaleOneLinearLocModel,
    NonStationaryTwoLinearShapeOneLinearLocModel,
    NonStationaryTwoLinearScaleOneLinearLocAndShapeModel,
    NonStationaryTwoLinearShapeOneLinearLocAndScaleModel,
    NonStationaryTwoLinearScaleAndShapeOneLinearLocModel,

    # Models with linear location parameter with double linearity
    NonStationaryTwoLinearLocationModel,
    # Simple linearity for the others
    NonStationaryTwoLinearLocationOneLinearScaleModel,
    NonStationaryTwoLinearLocationOneLinearShapeModel,
    NonStationaryTwoLinearLocationOneLinearScaleAndShapeModel,
    # Double Linearity for the others
    NonStationaryTwoLinearLocationAndScaleModel,
    NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel,
    NonStationaryTwoLinearLocationAndShape,
    NonStationaryTwoLinearLocationAndShapeOneLinearScaleModel,
    NonStationaryTwoLinearLocationAndScaleAndShapeModel,

]

if __name__ == '__main__':
    print(len(set(SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE)))
