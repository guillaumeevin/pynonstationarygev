import os.path as op

import matplotlib

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects, load_combination_name_for_tuple
from projects.projected_extreme_snowfall.results.experiment.model_as_truth_experiment import ModelAsTruthExperiment
from projects.projected_extreme_snowfall.results.part_2.average_bias import plot_average_bias, load_study, \
    compute_average_bias, plot_bias, plot_time_series
from projects.projected_extreme_snowfall.results.part_2.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.part_2.v2.utils import update_csv, is_already_done, load_excel, \
    main_sheet_name
from projects.projected_extreme_snowfall.results.setting_utils import set_up_and_load


def main_preliminary_projections():
    # Load parameters
    show = None
    fast = False
    snowfall = False

    matplotlib.use('Agg')
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)
    display_only_model_that_pass_gof_test = False
    # Load study
    model_classes = [NonStationaryLocationAndScaleAndShapeTemporalModel]
    massif_name = massif_names[0]
    altitudes = altitudes_list[0]
    altitude = altitudes[0]
    gcm_rcm_couple_to_study, safran_study = load_study(altitude, gcm_rcm_couples, safran_study_class, scenario,
                                                       study_class)
    print('number of couples loaded:', len(gcm_rcm_couple_to_study))

    average_bias, _ = compute_average_bias(gcm_rcm_couple_to_study, massif_name, safran_study, show=show)
    print('average bias for safran:', average_bias)
    if fast in [True]:
        alpha = ''
        gcm_rcm_couples_sampled_for_experiment = [('CNRM-CM5', 'ALADIN63')]
        gcm_rcm_couple_to_average_bias, gcm_rcm_couple_to_gcm_rcm_couple_to_biases = None, None
        # gcm_rcm_couples_sampled_for_experiment = [('NorESM1-M', 'REMO2015'), ('MPI-ESM-LR', 'REMO2009')]
    else:
        alpha = 30 if snowfall else 5
        gcm_rcm_couples_sampled_for_experiment, gcm_rcm_couple_to_average_bias, gcm_rcm_couple_to_gcm_rcm_couple_to_biases = plot_average_bias(gcm_rcm_couple_to_study, massif_name, average_bias,
                                                                   alpha, show=show)

    print(gcm_rcm_couples_sampled_for_experiment)

    study = 'snowfall' if snowfall else 'snow_load'
    csv_filename = 'last_{}_fast_{}_altitudes_{}_nb_of_models_{}_nb_gcm_rcm_couples_{}_alpha_{}.xlsx'.format(study, fast, altitude,
                                                                                                             len(model_classes),
                                                                                                             len(gcm_rcm_couple_to_study),
                                                                                                             alpha
                                                                                                                        )
    print(csv_filename)

    if gcm_rcm_couple_to_average_bias is not None:
        for couple in gcm_rcm_couples_sampled_for_experiment:
            plot_bias(gcm_rcm_couple_to_study[couple], gcm_rcm_couple_to_average_bias[couple], gcm_rcm_couple_to_gcm_rcm_couple_to_biases[couple], show)


    excel_filepath = op.join(CSV_PATH, csv_filename)

    print("Number of couples:", len(gcm_rcm_couples_sampled_for_experiment))

    couple_list = [(0, 0)]
    for index_name in [1, 2, 4, 5]:
        couple_list.extend([(index_name, index_name), (index_name, 0), (0, index_name)][:1])
    for index_name, j in couple_list[::1]:
        print(index_name, j)
        for gcm_rcm_couple in gcm_rcm_couples_sampled_for_experiment:
            combination = (index_name, j, 0)
            param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                combination)

            combination_name = load_combination_name_for_tuple(combination)
            xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                        scenario=scenario,
                                        selection_method_names=['aic'],
                                        model_classes=model_classes,
                                        massif_names=massif_names,
                                        fit_method=MarginFitMethod.evgam,
                                        temporal_covariate_for_fit=temporal_covariate_for_fit,
                                        remove_physically_implausible_models=remove_physically_implausible_models,
                                        display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                        gcm_rcm_couples_sampled_for_experiment=gcm_rcm_couples_sampled_for_experiment,
                                        param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                        )
            # plot time series
            gcm_rcm_couple_to_studies = xp.load_gcm_rcm_couple_to_studies(gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple)
            gcm_rcm_couple_to_study = {c: studies.study for c, studies in gcm_rcm_couple_to_studies.items()}
            gcm_rcm_couple_to_other_study = {c: s for c, s in gcm_rcm_couple_to_study.items() if c != gcm_rcm_couple}
            plot_time_series(massif_name, gcm_rcm_couple_to_study[gcm_rcm_couple], gcm_rcm_couple_to_other_study, show)

            if is_already_done(excel_filepath, combination_name, altitude, gcm_rcm_couple):
                continue
            nllh_list = xp.run_one_experiment(gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple)
            update_csv(excel_filepath, combination_name, altitude, gcm_rcm_couple, nllh_list)
    # Plot the content of the final df
    df = load_excel(excel_filepath, main_sheet_name)
    df = df.reindex(sorted(df.columns), axis=1)
    j_to_argmax = {j: int(df.iloc[:, j].values.argmax()) for j in range(len(df.columns))}
    for i, (index_name, row) in enumerate(df.iterrows()):
        print(index_name)
        values = [str(round(e, 1)) for e in list(row.values)]
        values = ['\\textbf{' + e + '}' if j_to_argmax[j] == i else e for j, e in enumerate(values)]
        print(' & '.join(values) + ' \\\\')


if __name__ == '__main__':
    main_preliminary_projections()
