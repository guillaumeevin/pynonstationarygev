import unittest
import numpy as np
from datetime import datetime

import pandas as pd

from extreme_data.edf_data.weather_types import load_df_weather_types, wp_int_to_wp_str, ANTICYCLONIC, STEADY_OCEANIC
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSwe3Days
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranTemperature, SafranPrecipitation1Day
from extreme_data.meteo_france_data.scm_models_data.utils import date_to_str


class TestWeatherTypes(unittest.TestCase):

    def test_df_weather_types(self):
        df = load_df_weather_types()
        self.assertEqual(len(df), 20354)
        # Assert values
        self.assertEqual(df.iloc[0, :].values[0], wp_int_to_wp_str[5])
        self.assertEqual(df.iloc[-1, :].values[0], wp_int_to_wp_str[8])
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
        wp_str = [wp_int_to_wp_str[wp_int] for wp_int in range(1, 9)]
        wp_to_expected_percentages = dict(zip(wp_str, expected_percentages))
        # Compute percentages
        df = load_df_weather_types()
        wp_to_found_percentages = 100 * df['WP'].value_counts() / len(df)
        wp_to_found_percentages = {k: round(v) for k, v in wp_to_found_percentages.to_dict().items()}
        # They remove one the wp1 so that the sum of the percentages sum to 100
        wp_to_found_percentages[wp_int_to_wp_str[1]] -= 1
        self.assertEqual(sum(wp_to_found_percentages.values()), 100)
        # wp_to_found_percentages = wp_to_found_percentages.astype(int)
        self.assertEqual(wp_to_expected_percentages, wp_to_found_percentages)

    def test_anticyclonique_weather_pattern(self):
        study = SafranPrecipitation1Day(altitude=900, year_min=1954, year_max=2008)
        no_rain = []
        rain = []
        for year, wps in study.year_to_wps.items():
            daily_time_serie_array = study.year_to_daily_time_serie_array[year]
            self.assertEqual(len(daily_time_serie_array), len(wps))
            mask = np.array(wps) == ANTICYCLONIC
            no_rain.extend(np.max(daily_time_serie_array[mask, :], axis=1))
            rain.extend(np.max(daily_time_serie_array[~mask, :], axis=1))
        # For 90% of the anticyclonic days, the daily max precipitation (snowfall + rainfall) for a massifs is < 0.2mm
        # Valid that the anticyclonic days seems to well defined (but with a big variety still...)
        self.assertLess(np.quantile(no_rain, 0.5), 0.2)
        self.assertLess(5, np.quantile(rain, 0.5))

    def test_weather_patterns_maxima(self):
        study = SafranPrecipitation1Day(altitude=900, year_min=1954, year_max=2008)
        s = pd.Series(np.concatenate([v for v in study.year_to_wp_for_annual_maxima.values()]))
        storms_ranking = s.value_counts()
        self.assertEqual(storms_ranking.index[0], STEADY_OCEANIC)
        self.assertEqual(storms_ranking.index[-1], ANTICYCLONIC)
        self.assertEqual(storms_ranking.values[0], 376)
        self.assertEqual(storms_ranking.values[-1], 9)


if __name__ == '__main__':
    unittest.main()
