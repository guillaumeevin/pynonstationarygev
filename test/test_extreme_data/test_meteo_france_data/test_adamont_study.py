import unittest

from extreme_data.meteo_france_data.adamont_data.snowfall_simulation import SafranSnowfallSimulationRCP85


class TestAdamontStudy(unittest.TestCase):

    def test_year_to_date(self):

        study = SafranSnowfallSimulationRCP85(altitude=900)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()