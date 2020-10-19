import os.path as op
import numpy as np
import unittest
from random import sample

import pandas as pd

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days
from extreme_data.meteo_france_data.scm_models_data.safran.cumulated_study import NB_DAYS
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, SafranTemperature, \
    SafranPrecipitation, SafranSnowfall3Days, SafranRainfall3Days, SafranNormalizedPreciptationRateOnWetDays
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDIES, ALL_ALTITUDES, SCM_STUDY_CLASS_TO_ABBREVIATION
from root_utils import get_display_name_from_object_type


class TestSCMAllStudy(unittest.TestCase):

    def test_year_to_date(self):
        year = 2019
        study = SafranSnowfall(altitude=900, year_min=year, year_max=year)
        first_day, *_, last_day = study.year_to_days[year]
        self.assertEqual('{}-08-01'.format(year - 1), first_day)
        self.assertEqual('{}-07-31'.format(year), last_day)

    def test_year_to_winter_extended_date(self):
        year = 2019
        study = SafranSnowfall(altitude=900, year_min=year, year_max=year, season=Season.winter_extended)
        first_day, *_, last_day = study.year_to_days[year]
        self.assertEqual('{}-11-01'.format(year - 1), first_day)
        self.assertEqual('{}-05-31'.format(year), last_day)
        days = study.year_to_days[year]
        daily_time_series = study.year_to_daily_time_serie_array[year]
        self.assertEqual(len(days), len(daily_time_series))

    def test_study_for_split(self):
        split_years = [1959 + 10 * i for i in range(7)]
        study = SafranSnowfall(altitude=900, split_years=split_years)
        self.assertEqual(split_years, list(study.ordered_years))

    def test_instantiate_studies(self):
        study_classes = SCM_STUDIES
        nb_sample = 2
        for nb_days in sample(set(NB_DAYS), k=nb_sample):
            for study in study_iterator_global(study_classes=study_classes,
                                               only_first_one=False, verbose=False,
                                               altitudes=sample(set(ALL_ALTITUDES), k=nb_sample), nb_days=nb_days):
                first_path_file = study.ordered_years_and_path_files[0][0]
                variable_object = study.load_variable_object(path_file=first_path_file)
                self.assertEqual((365, 263), variable_object.daily_time_serie_array.shape,
                                 msg='{} days for type {}'.format(nb_days, get_display_name_from_object_type(
                                     type(variable_object))))

    def test_instantiate_studies_with_number_of_days(self):
        altitude = 900
        year_min = 1959
        year_max = 2000
        study_classes = [SafranSnowfall3Days, SafranRainfall3Days, CrocusSnowLoad3Days]
        for study_class in study_classes:
            study_class(altitude=altitude, year_min=year_min, year_max=year_max)

    def test_variables(self):
        for study_class in SCM_STUDY_CLASS_TO_ABBREVIATION.keys():
            study = study_class(year_max=1959)
            _ = study.year_to_annual_maxima[1959]
        self.assertTrue(True)


class TestSCMSafranNormalizedPrecipitationRateOnWetDays(unittest.TestCase):

    def test_annual_maxima(self):
        study = SafranNormalizedPreciptationRateOnWetDays(year_max=1960)
        self.assertFalse(np.isnan(study.year_to_annual_maxima[1959]).any())


class TestSCMStudy(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.study = None

    def check(self, massif_name_to_value_to_check):
        df_annual_total = self.study.df_annual_total
        for massif_name, value in massif_name_to_value_to_check.items():
            found_value = df_annual_total.loc[:, massif_name].mean()
            self.assertEqual(value, self.round(found_value))

    def round(self, f):
        raise NotImplementedError


class TestSCMSafranSnowfall(TestSCMStudy):

    def setUp(self) -> None:
        super().setUp()
        self.study = SafranSnowfall()

    def test_massif_safran(self):
        df_centroid = pd.read_csv(op.join(self.study.map_full_path, 'coordonnees_massifs_alpes.csv'))
        # Assert that the massif names are the same between SAFRAN and the coordinate file
        assert not set(self.study.study_massif_names).symmetric_difference(set(df_centroid['NOM']))

    def test_all_data(self):
        all_daily_series = self.study.all_daily_series
        self.assertEqual(all_daily_series.ndim, 2)
        self.assertEqual(all_daily_series.shape[1], 23)
        self.assertEqual(all_daily_series.shape[0], 22280)


class TestSCMPrecipitation(TestSCMStudy):

    def setUp(self) -> None:
        super().setUp()
        self.study = SafranPrecipitation(altitude=1800, year_min=1959, year_max=2002, nb_consecutive_days=1)

    def test_durand(self):
        # Test based on Durand paper
        # (some small differences probably due to the fact that SAFRAN model has evolved since then)
        # Test for the mean total precipitation (rainfall + snowfall) between 1958 and 2002
        self.check({
            "Mercantour": 1300,
            'Chablais': 1947,
        })

    def round(self, f):
        return int(f)


class TestSafranTemperature(TestSCMStudy):

    def setUp(self):
        super().setUp()
        self.study = SafranTemperature(altitude=1800, year_min=1959, year_max=2002)

    def test_durand(self):
        # Test based on Durand paper
        # Test for the mean temperature between 1958 and 2002
        self.check({
            "Mercantour": 5.3,
            'Chablais': 3.5,
        })

    def round(self, f):
        return round(float(f), 1)


if __name__ == '__main__':
    unittest.main()
