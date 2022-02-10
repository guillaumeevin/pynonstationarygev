import matplotlib

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel, NonStationaryThreeLinearLocationAndScaleAndShapeModel, \
    NonStationaryFourLinearLocationAndScaleAndShapeModel, NonStationaryFiveLinearLocationAndScaleAndShapeModel, \
    NonStationarySevenLinearLocationAndScaleAndShapeModel, \
    NonStationaryEightLinearLocationAndScaleAndShapeModel
from projected_extremes.section_results.utils.average_bias import compute_average_bias, load_study, plot_average_bias
from projected_extremes.section_results.utils.setting_utils import set_up_and_load
from projected_extremes.section_results.validation_experiment.model_as_truth_experiment import ModelAsTruthExperiment


def main_preliminary_projections():
    # Load parameters
    show = False
    # print('sleeping...')
    # time.sleep(60*30)

    fast = True
    snowfall = False

    if show in [None, True]:
        matplotlib.use('Agg')
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    year_max_for_pseudo_obs, year_max_for_gcm = 2019, 2100

    # for percentage in [0.7]:
    # year_max_for_pseudo_obs, year_max_for_gcm = get_last_year_for_the_train_set(percentage), 2019
    weight_on_observation = 1
    print('weight on observation=', weight_on_observation)

    linear_effects = (False, False, False)

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall)

    altitudes_list = [[2100], [2400], [2700], [3000], [3300], [3600]][:1]
    altitudes_list = [[900], [1200], [1500], [1800]][3:4]
    print(altitudes_list)
    model_classes_list = [NonStationaryLocationAndScaleAndShapeTemporalModel,
                          NonStationaryTwoLinearLocationAndScaleAndShapeModel,
                          NonStationaryThreeLinearLocationAndScaleAndShapeModel,
                          NonStationaryFourLinearLocationAndScaleAndShapeModel][:]

    for model_class in model_classes_list:
        model_classes = [model_class]
        for altitudes in altitudes_list:
            run_mas(altitudes, display_only_model_that_pass_gof_test, fast, gcm_rcm_couples, massif_names,
                    model_classes, remove_physically_implausible_models, safran_study_class, scenario, show, snowfall,
                    study_class, temporal_covariate_for_fit, year_max_for_gcm, year_max_for_pseudo_obs,
                    weight_on_observation, linear_effects, fit_method, season)


def run_mas(altitudes, display_only_model_that_pass_gof_test, fast, gcm_rcm_couples, massif_names,
            model_classes, remove_physically_implausible_models, safran_study_class, scenario, show, snowfall,
            study_class, temporal_covariate_for_fit, year_max_for_gcm, year_max_for_pseudo_obs, weight_on_observation,
            linear_effects, fit_method, season):
    altitude = altitudes[0]
    print('Altitude={}'.format(altitude))
    gcm_rcm_couple_to_study, safran_study = load_study(altitude, gcm_rcm_couples, safran_study_class, scenario,
                                                       study_class, season)

    for massif_name in massif_names[::1]:
        if massif_name in safran_study.study_massif_names:
            print(massif_name)

            average_bias, _ = compute_average_bias(gcm_rcm_couple_to_study, massif_name, safran_study, show=show)
            gcm_rcm_couples_sampled_for_experiment, gcm_rcm_couple_to_average_bias, gcm_rcm_couple_to_gcm_rcm_couple_to_biases = plot_average_bias(
                gcm_rcm_couple_to_study, massif_name, average_bias,
                alpha=1000, show=show)

            print("Number of couples:", len(gcm_rcm_couples_sampled_for_experiment))
            for i in [-1, 0, 1, 2, 4, 5][1:2]:
                # for i in [-1, 0, 5][:]:
                print('parametrization', i)
                for gcm_rcm_couple in gcm_rcm_couples_sampled_for_experiment:
                    combination = (i, i, 0)

                    xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples,
                                                safran_study_class,
                                                study_class, season=season,
                                                scenario=scenario,
                                                selection_method_names=['aic'],
                                                model_classes=model_classes,
                                                massif_names=[massif_name],
                                                fit_method=fit_method,
                                                temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                remove_physically_implausible_models=remove_physically_implausible_models,
                                                display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                                gcm_rcm_couples_sampled_for_experiment=gcm_rcm_couples_sampled_for_experiment,
                                                combination=combination,
                                                year_max_for_gcm=year_max_for_gcm,
                                                year_max_for_pseudo_obs=year_max_for_pseudo_obs,
                                                weight_on_observation=weight_on_observation,
                                                linear_effects=linear_effects,
                                                )
                    xp.run_one_experiment(gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple)


if __name__ == '__main__':
    main_preliminary_projections()
