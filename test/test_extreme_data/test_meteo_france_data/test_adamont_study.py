import unittest

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, gcm_rcm_couple_to_full_name
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall


class TestAdamontStudy(unittest.TestCase):

    def test_load_adamont_snowfall(self):
        study_list = [
            AdamontSnowfall(altitude=900),
            AdamontSnowfall(altitude=1800)
        ]
        study_list.extend([AdamontSnowfall(altitude=900, scenario=AdamontScenario.rcp45),
                           AdamontSnowfall(altitude=900, scenario=AdamontScenario.rcp85)])
        study_list.extend([AdamontSnowfall(altitude=900, gcm_rcm_couple=gcm_rcm_couple)
                           for gcm_rcm_couple in gcm_rcm_couple_to_full_name.keys()])
        for study in study_list:
            annual_maxima_for_year_min = study.year_to_annual_maxima[study.year_min]
            # print(study.altitude, study.scenario_name, study.gcm_rcm_couple)
            # print(annual_maxima_for_year_min)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
