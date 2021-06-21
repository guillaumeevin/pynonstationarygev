import unittest

from extreme_data.meteo_france_data.mean_alps_temperature import load_year_to_mean_alps_temperatures


class TestMeanGlobalTemperatures(unittest.TestCase):

    def test_year_to_mean_alps_temperatures(self):
        d = load_year_to_mean_alps_temperatures()
        self.assertIn(2019, d)
        self.assertNotIn(2020, d)
        self.assertIn(1959, d)
        key = list(d.keys())[0]
        self.assertIsInstance(key, float)
        value = list(d.values())[0]
        self.assertIsInstance(value, float)


if __name__ == '__main__':
    unittest.main()
