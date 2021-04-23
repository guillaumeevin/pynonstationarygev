import datetime
import time

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.part_1.model_as_truth_experiment import ModelAsTruthExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load


def main_model_as_truth_experiment():
    start = time.time()
    fast = True

    altitudes_list, climate_coordinates_with_effects_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit = set_up_and_load(fast)

    for altitudes in altitudes_list[:1]:
        for climate_coordinates_with_effects in climate_coordinates_with_effects_list:
            xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                        scenario=scenario,
                                        model_classes=model_classes,
                                        massif_names=massif_names,
                                        fit_method=MarginFitMethod.evgam,
                                        temporal_covariate_for_fit=temporal_covariate_for_fit,
                                        remove_physically_implausible_models=True,
                                        display_only_model_that_pass_gof_test=True,
                                        climate_coordinates_with_effects=climate_coordinates_with_effects)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_model_as_truth_experiment()
