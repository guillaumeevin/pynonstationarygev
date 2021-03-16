from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from projects.projected_swe.model_as_truth_visualizer.model_as_truth import ModelAsTruth
from projects.projected_swe.weight_solver.default_weight_solver import EqualWeight
from projects.projected_swe.weight_solver.indicator import AnnualMaximaMeanIndicator, ReturnLevel30YearsIndicator
from projects.projected_swe.weight_solver.knutti_weight_solver import KnuttiWeightSolver
from projects.projected_swe.weight_solver.knutti_weight_solver_with_bootstrap import \
    KnuttiWeightSolverWithBootstrapVersion1, KnuttiWeightSolverWithBootstrapVersion2


def main():
    altitude = 900
    year_min_histo = 1982
    year_max_histo = 2011
    year_min_projected = 2070
    year_max_projected = 2099
    scenario = AdamontScenario.rcp85_extended
    fast = None
    gcm_rcm_couples = get_gcm_rcm_couples(adamont_scenario=scenario)

    if fast is None:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = None
        knutti_weight_solver_classes = [EqualWeight,
                                        KnuttiWeightSolver,
                                        KnuttiWeightSolverWithBootstrapVersion1,
                                        KnuttiWeightSolverWithBootstrapVersion2]
        indicator_class = ReturnLevel30YearsIndicator
        gcm_rcm_couples = gcm_rcm_couples[:3]
        sigma_list = [10, 100, 1000, 10000]

    elif fast:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = ['Vercors']

        knutti_weight_solver_classes = [EqualWeight, KnuttiWeightSolver,
                                        KnuttiWeightSolverWithBootstrapVersion1]
        indicator_class = ReturnLevel30YearsIndicator
        gcm_rcm_couples = gcm_rcm_couples[:3]
        sigma_list = [100, 1000]

    else:
        massif_names = None
        indicator_class = AnnualMaximaMeanIndicator
        knutti_weight_solver_classes = [KnuttiWeightSolver,
                                        KnuttiWeightSolverWithBootstrapVersion1,
                                        KnuttiWeightSolverWithBootstrapVersion2]

    observation_study = SafranSnowfall2020(altitude=altitude, year_min=year_min_histo, year_max=year_max_histo)
    couple_to_historical_study = {c: AdamontSnowfall(altitude=altitude, scenario=scenario,
                                                     year_min=year_min_histo, year_max=year_max_histo,
                                                     gcm_rcm_couple=c) for c in gcm_rcm_couples}
    couple_to_projected_study = {c: AdamontSnowfall(altitude=altitude, scenario=scenario,
                                                    year_min=year_min_projected, year_max=year_max_projected,
                                                    gcm_rcm_couple=c) for c in gcm_rcm_couples
                                 }

    model_as_truth = ModelAsTruth(observation_study, couple_to_projected_study, couple_to_historical_study,
                                  indicator_class, knutti_weight_solver_classes, massif_names,
                                  add_interdependence_weight=False)
    model_as_truth.plot_against_sigma(sigma_list)


if __name__ == '__main__':
    main()
