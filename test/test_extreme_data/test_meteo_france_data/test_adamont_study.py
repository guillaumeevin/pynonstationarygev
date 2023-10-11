import unittest

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_crocus import AdamontSwe, AdamontSnowLoad
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import get_gcm_rcm_couple_adamont_to_full_name
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, adamont_scenarios_real, \
    rcp_scenarios, get_gcm_rcm_couples, rcm_scenarios_extended
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusVariable
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day


class TestAdamontStudy(unittest.TestCase):

    def test_load_adamont_snowfall(self):
        for version in [1, 2][:]:
            self.load_many_study(AdamontSnowfall, version)
        self.assertTrue(True)

    def test_load_adamont_swe(self):
        self.load_many_study(AdamontSwe, version=2, load_index=True)
        self.assertTrue(True)

    def test_load_adamont_snow_load(self):
        maxima = [study_class(altitude=1800, gcm_rcm_couple=('HadGEM2-ES', 'RACMO22E'),
                              scenario=AdamontScenario.rcp85_extended).year_to_annual_maxima[2000][0]
                  for study_class in [AdamontSwe, AdamontSnowLoad]]
        swe, snow_load = maxima
        snow_load_from_swe = swe * CrocusVariable.snow_load_multiplication_factor
        self.assertEqual(snow_load_from_swe, snow_load)

    def load_many_study(self, adamont_study_class, version, load_index=False):
        study_list = [
            adamont_study_class(altitude=900),
            adamont_study_class(altitude=1800)
        ]
        for scenario in rcp_scenarios + rcm_scenarios_extended:
            gcm_rcm_couples = get_gcm_rcm_couples(scenario, version)
            if len(gcm_rcm_couples) > 0:
                first_gcm_rcm_couple = gcm_rcm_couples[0]
                study_list.append(adamont_study_class(altitude=900, scenario=scenario,
                                                      gcm_rcm_couple=first_gcm_rcm_couple))
        study_list.extend([adamont_study_class(altitude=900, gcm_rcm_couple=gcm_rcm_couple)
                           for gcm_rcm_couple in get_gcm_rcm_couples()])
        for study in study_list:
            _ = study.year_to_annual_maxima[study.year_min]
            if load_index:
                _ = study.year_to_annual_maxima_index[study.year_min]

    def test_massifs_names_adamont_v2(self):
        year_min = 2004
        for altitude in [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]:
            reanalysis_study = SafranSnowfall1Day(altitude=altitude, year_min=year_min)
            for gcm_rcm_couple in get_gcm_rcm_couple_adamont_to_full_name().keys():
                adamont_study = AdamontSnowfall(altitude=altitude,
                                                year_min=year_min, gcm_rcm_couple=gcm_rcm_couple)
                assert set(adamont_study.study_massif_names) == set(reanalysis_study.study_massif_names)

    def test_rcp_extended(self):
        for version in [1, 2]:
            study = AdamontSnowfall(altitude=1800, gcm_rcm_couple=('HadGEM2-ES', 'RACMO22E'),
                                    scenario=AdamontScenario.rcp85_extended)
            self.assertEqual(len(study.ordered_years), len(study.massif_name_to_annual_maxima["Vanoise"]))
        self.assertTrue(True)

    def test_existing_gcm_rcm_couple_and_rcp(self):
        altitude = 1800
        for scenario in rcp_scenarios[:]:
            l = []
            for gcm_rcm_couple in get_gcm_rcm_couples(scenario):
                adamont_study = AdamontSnowfall(altitude=altitude,
                                                year_min=2098, gcm_rcm_couple=gcm_rcm_couple,
                                                scenario=scenario)
                try:
                    _ = adamont_study.year_to_annual_maxima[2098]
                except FileNotFoundError:
                    l.append(gcm_rcm_couple)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
