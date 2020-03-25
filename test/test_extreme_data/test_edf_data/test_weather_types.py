import unittest
from datetime import datetime

import pandas as pd

from extreme_data.edf_data.weather_types import load_df_weather_types
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSwe3Days
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranTemperature
from extreme_data.meteo_france_data.scm_models_data.utils import date_to_str


class TestWeatherTypes(unittest.TestCase):

    def test_df_weather_types(self):
        df = load_df_weather_types()
        self.assertEqual(len(df), 20354)
        # Assert values
        self.assertEqual(df.iloc[0, :].values[0], 5)
        self.assertEqual(df.iloc[-1, :].values[0], 8)
        # Assert keys
        self.assertEqual(date_to_str(datetime(year=1953, month=1, day=1)), df.index[0])
        self.assertEqual(date_to_str(datetime(year=2008, month=9, day=22)), df.index[-1])

    def test_assertion_wps(self):
        with self.assertRaises(AssertionError):
            print(CrocusSwe3Days(altitude=900, year_max=2020).year_to_wps)
        with self.assertRaises(AssertionError):
            print(CrocusSwe3Days(altitude=900, year_min=1952).year_to_wps)
        study = CrocusSwe3Days(altitude=900, year_min=1954, year_max=2008)
        d = study.year_to_wps
        self.assertTrue(True)

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

    def test_anticyclonique_weather_pattern(self):
        study = CrocusSwe3Days(altitude=900, year_min=1954, year_max=2008)
        pass



if __name__ == '__main__':
    unittest.main()
