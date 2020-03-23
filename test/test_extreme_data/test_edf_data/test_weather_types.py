import unittest

from extreme_data.edf_data.weather_types import load_df_weather_types
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranTemperature


class TestWeatherTypes(unittest.TestCase):

    def test_df_weather_types(self):
        df = load_df_weather_types()
        self.assertEqual(len(df), 20354)
        first = df.iloc[0].values[0]
        last = df.iloc[-1].values[0]
        self.assertEqual(first, 5)
        self.assertEqual(last, 8)


if __name__ == '__main__':
    unittest.main()
