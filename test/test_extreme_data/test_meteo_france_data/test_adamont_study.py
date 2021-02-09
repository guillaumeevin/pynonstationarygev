import unittest

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import get_gcm_rcm_couple_adamont_to_full_name
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day


class TestAdamontStudy(unittest.TestCase):

    def test_load_adamont_snowfall(self):
        for version in [1, 2]:
            study_list = [
                AdamontSnowfall(altitude=900, adamont_version=version),
                AdamontSnowfall(altitude=1800, adamont_version=version)
            ]
            study_list.extend([
                AdamontSnowfall(altitude=900, scenario=AdamontScenario.rcp45, adamont_version=version),
                AdamontSnowfall(altitude=900, scenario=AdamontScenario.rcp85, adamont_version=version),
                AdamontSnowfall(altitude=900, scenario=AdamontScenario.rcp85_extended, adamont_version=version)
            ])
            gcm_rcm_couple_to_full_name = get_gcm_rcm_couple_adamont_to_full_name(version)
            study_list.extend([AdamontSnowfall(altitude=900, gcm_rcm_couple=gcm_rcm_couple, adamont_version=version)
                               for gcm_rcm_couple in gcm_rcm_couple_to_full_name.keys()])
            for study in study_list:
                annual_maxima_for_year_min = study.year_to_annual_maxima[study.year_min]
                # print(study.altitude, study.scenario_name, study.gcm_rcm_couple)
                # print(len(study.massif_name_to_annual_maxima['Vanoise']))
            self.assertTrue(True)

    def test_massifs_names_adamont_v2(self):
        year_min = 2004
        adamont_version = 2 # this test will not pass with adamont version 1
        for altitude in [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]:
            reanalysis_study = SafranSnowfall1Day(altitude=altitude, year_min=year_min)
            for gcm_rcm_couple in get_gcm_rcm_couple_adamont_to_full_name(adamont_version).keys():
                adamont_study = AdamontSnowfall(altitude=altitude, adamont_version=adamont_version,
                                                year_min=year_min, gcm_rcm_couple=gcm_rcm_couple)
                assert set(adamont_study.study_massif_names) == set(reanalysis_study.study_massif_names)


if __name__ == '__main__':
    unittest.main()
