import unittest

import pandas as pd

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

    def test_yearly_percentages(self):
        # Expected Percentages come from the original paper
        expected_percentages = [7, 23, 8, 18, 7, 6, 3, 28]
        wp_ids = list(range(1, 9))
        wp_to_expected_percentages = dict(zip(wp_ids, expected_percentages))
        # Compute percentages
        df = load_df_weather_types()
        wp_to_found_percentages = 100 * df['WP'].value_counts() / len(df)
        wp_to_found_percentages = {int(k): round(v) for k, v in wp_to_found_percentages.to_dict().items()}
        # They remove one the wp1 so that the sum of the percentages sum to 100
        wp_to_found_percentages[1] -= 1
        self.assertEqual(sum(wp_to_found_percentages.values()), 100)
        # wp_to_found_percentages = wp_to_found_percentages.astype(int)
        self.assertEqual(wp_to_expected_percentages, wp_to_found_percentages)


if __name__ == '__main__':
    unittest.main()
