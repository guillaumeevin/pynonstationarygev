import unittest
import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020
from extreme_data.meteo_france_data.scm_models_data.utils_function import ReturnLevelBootstrap
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_fit.model.utils import set_seed_for_test
from projects.archive.projected_swe.weight_solver.indicator import AnnualMaximaMeanIndicator, ReturnLevel30YearsIndicator
from projects.archive.projected_swe.weight_solver.knutti_weight_solver import KnuttiWeightSolver
from projects.archive.projected_swe.weight_solver.knutti_weight_solver_with_bootstrap import \
    KnuttiWeightSolverWithBootstrapVersion1, KnuttiWeightSolverWithBootstrapVersion2


class TestModelAsTruth(unittest.TestCase):

    def test_knutti_weight_solver(self):
        set_seed_for_test()
        ReturnLevelBootstrap.only_physically_plausible_fits = True
        altitude = 900
        year_min = 1982
        year_max = 2011
        scenario = AdamontScenario.rcp85_extended
        observation_study = SafranSnowfall2020(altitude=altitude, year_min=year_min, year_max=year_max)
        couple_to_study = {c: AdamontSnowfall(altitude=altitude, scenario=scenario,
                                              year_min=year_min, year_max=year_max,
                                              gcm_rcm_couple=c) for c in get_gcm_rcm_couples(adamont_scenario=scenario)}
        massif_names = ['Vercors']
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        for knutti_weight_solver_class in [KnuttiWeightSolver,
                                           KnuttiWeightSolverWithBootstrapVersion1,
                                           KnuttiWeightSolverWithBootstrapVersion2][:]:
            if knutti_weight_solver_class in [KnuttiWeightSolverWithBootstrapVersion1, KnuttiWeightSolverWithBootstrapVersion2]:
                idx = 1
                sigma = 1000
            else:
                sigma = 10
                idx = 0
            for indicator_class in [AnnualMaximaMeanIndicator, ReturnLevel30YearsIndicator][idx:]:
                for add_interdependence_weight in [False, True]:
                    knutti_weight = knutti_weight_solver_class(sigma_skill=sigma, sigma_interdependence=sigma,
                                                               massif_names=massif_names,
                                                               observation_study=observation_study,
                                                               couple_to_historical_study=couple_to_study,
                                                               indicator_class=indicator_class,
                                                               add_interdependence_weight=add_interdependence_weight
                                                               )
                    # print(knutti_weight.couple_to_weight)
                    weight = knutti_weight.couple_to_weight[('CNRM-CM5', 'CCLM4-8-17')]
                    self.assertFalse(np.isnan(weight))

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
