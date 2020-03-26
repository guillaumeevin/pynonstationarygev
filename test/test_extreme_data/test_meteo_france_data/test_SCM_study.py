import os.path as op
import unittest
from random import sample

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.safran.cumulated_study import NB_DAYS
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, SafranTemperature, \
    SafranPrecipitation
from extreme_data.meteo_france_data.scm_models_data.utils import SeasonForTheMaxima
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDIES, ALL_ALTITUDES
from root_utils import get_display_name_from_object_type


class TestSCMAllStudy(unittest.TestCase):

    def test_year_to_date(self):
        year = 2019
        study = SafranSnowfall(altitude=900, year_min=year, year_max=year)
        first_day, *_, last_day = study.year_to_days[year]
        self.assertEqual('{}-08-01'.format(year - 1), first_day)
        self.assertEqual('{}-07-31'.format(year), last_day)

    def test_year_to_winter_date(self):
        year = 2019
        study = SafranSnowfall(altitude=900, year_min=year, year_max=year, season=SeasonForTheMaxima.winter_extended)
        first_day, *_, last_day = study.year_to_days[year]
        self.assertEqual('{}-11-01'.format(year - 1), first_day)
        self.assertEqual('{}-05-31'.format(year), last_day)
        days = study.year_to_days[year]
        daily_time_series = study.year_to_daily_time_serie_array[year]
        self.assertEqual(len(days), len(daily_time_series))

    def test_instantiate_studies(self):
        nb_sample = 2
        for nb_days in sample(set(NB_DAYS), k=nb_sample):
            for study in study_iterator_global(study_classes=SCM_STUDIES,
                                               only_first_one=False, verbose=False,
                                               altitudes=sample(set(ALL_ALTITUDES), k=nb_sample), nb_days=nb_days):
                first_path_file = study.ordered_years_and_path_files[0][0]
                variable_object = study.load_variable_object(path_file=first_path_file)
                self.assertEqual((365, 263), variable_object.daily_time_serie_array.shape,
                                 msg='{} days for type {}'.format(nb_days, get_display_name_from_object_type(
                                     type(variable_object))))


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
