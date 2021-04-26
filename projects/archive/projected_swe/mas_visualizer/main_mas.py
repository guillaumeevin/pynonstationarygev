from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020
from extreme_data.meteo_france_data.scm_models_data.utils_function import ReturnLevelBootstrap
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from projects.archive.projected_swe.mas_visualizer import ModelAsTruth
from projects.archive.projected_swe.weight_solver.indicator import AnnualMaximaMeanIndicator
from projects.archive.projected_swe.weight_solver import KnuttiWeightSolver
from projects.archive.projected_swe.weight_solver.knutti_weight_solver_with_bootstrap import \
    KnuttiWeightSolverWithBootstrapVersion1, KnuttiWeightSolverWithBootstrapVersion2


def main():
    # Set some parameters for the bootstrap
    ReturnLevelBootstrap.only_physically_plausible_fits = True
    year_min_histo = 1982
    year_max_histo = 2011
    scenario = AdamontScenario.rcp85_extended
    fast = False
    gcm_rcm_couples = get_gcm_rcm_couples(adamont_scenario=scenario)
    indicator_class = AnnualMaximaMeanIndicator

    if fast is None:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        year_couples = [(1982, 2011), (2012, 2041), (2042, 2071)]
        altitudes = [900, 1800]

        massif_names = None
        knutti_weight_solver_classes = [KnuttiWeightSolver,
                                        KnuttiWeightSolverWithBootstrapVersion1,
                                        KnuttiWeightSolverWithBootstrapVersion2][:1]
        gcm_rcm_couples = gcm_rcm_couples[:8]
        sigma_list = [6, 7, 8]

    elif fast:
        altitudes = [900]
        year_couples = [(1982, 2011)]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = ['Chartreuse']
        knutti_weight_solver_classes = [KnuttiWeightSolver]
        gcm_rcm_couples = gcm_rcm_couples[:3]
        sigma_list = [10]

    else:
        altitudes = [900, 1800, 2700, 3600][:]
        year_couples = [(1982, 2011), (2012, 2041), (2042, 2071), (2070, 2099)][:]
        massif_names = None
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        knutti_weight_solver_classes = [KnuttiWeightSolver][:]
        gcm_rcm_couples = gcm_rcm_couples[:]
        sigma_list = [i * 0.25 + 1 for i in range(13)]

    for altitude in altitudes:
            observation_study = SafranSnowfall2020(altitude=altitude, year_min=year_min_histo, year_max=year_max_histo)
            couple_to_historical_study = {c: AdamontSnowfall(altitude=altitude, scenario=scenario,
                                                             year_min=year_min_histo, year_max=year_max_histo,
                                                             gcm_rcm_couple=c) for c in gcm_rcm_couples}


            model_as_truth = ModelAsTruth(observation_study, couple_to_historical_study,
                                          indicator_class, knutti_weight_solver_classes, massif_names,
                                          add_interdependence_weight=False)

            for year_couple in year_couples:
                year_min_projected, year_max_projected = year_couple
                if (year_min_projected, year_max_projected) == (year_min_histo, year_max_histo):
                    couple_to_projected_study = couple_to_historical_study
                else:
                    couple_to_projected_study = {c: AdamontSnowfall(altitude=altitude, scenario=scenario,
                                                                    year_min=year_min_projected,
                                                                    year_max=year_max_projected,
                                                                    gcm_rcm_couple=c) for c in gcm_rcm_couples
                                                 }

                model_as_truth.plot_against_sigma(couple_to_projected_study, sigma_list)


if __name__ == '__main__':
    main()
