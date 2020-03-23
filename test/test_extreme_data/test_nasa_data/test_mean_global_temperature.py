import unittest

from extreme_data.edf_data.weather_types import load_df_weather_types
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranTemperature
from extreme_data.nasa_data.global_mean_temperature import load_year_to_mean_global_temperature


class TestMeanGlobalTemperatures(unittest.TestCase):

    def test_year_to_mean_global_temperature(self):
        d = load_year_to_mean_global_temperature()
        self.assertNotIn(2019, d)
        self.assertIn(2009, d)
        key = list(d.keys())[0]
        self.assertIsInstance(key, float)
        value = list(d.values())[0]
        self.assertIsInstance(value, float)

if __name__ == '__main__':
    unittest.main()
