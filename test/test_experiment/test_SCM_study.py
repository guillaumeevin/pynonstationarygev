import os.path as op
import unittest
from random import sample

import pandas as pd

from experiment.meteo_france_data.scm_models_data.cumulated_study import NB_DAYS
from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, ExtendedSafranSnowfall, \
    SafranTemperature, \
    SafranTotalPrecip
from experiment.meteo_france_data.visualization.study_visualization.main_study_visualizer import study_iterator, \
    study_iterator_global, SCM_STUDIES, ALL_ALTITUDES
from experiment.meteo_france_data.visualization.study_visualization.study_visualizer import StudyVisualizer
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest
from utils import get_display_name_from_object_type


class TestSCMAllStudy(unittest.TestCase):

    def test_extended_run(self):
        for study_class in [ExtendedSafranSnowfall]:
            for study in study_iterator(study_class, only_first_one=True, verbose=False):
                study_visualizer = StudyVisualizer(study, show=False, save_to_file=False, multiprocessing=True)
                study_visualizer.df_trend_spatio_temporal(GevLocationChangePointTest, [1958, 1959, 1960],
                                                          nb_massif_for_fast_mode=1)
        self.assertTrue(True)

    def test_instantiate_studies(self):
        nb_sample = 2
        for nb_days in sample(set(NB_DAYS), k=nb_sample):
            for study in study_iterator_global(study_classes=SCM_STUDIES,
                                               only_first_one=False, verbose=False,
                                               altitudes=sample(set(ALL_ALTITUDES), k=nb_sample), nb_days=nb_days):
                self.assertTrue('day' in study.variable_name)
                first_path_file = study.ordered_years_and_path_files[0][0]
                variable_array = study.load_variables(path_file=first_path_file)
                variable_object = study.instantiate_variable_object(variable_array)
                self.assertEqual((365, 263), variable_object.daily_time_serie_array.shape,
                                 msg='{} days for type {}'.format(nb_days, get_display_name_from_object_type(type(variable_object))))


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
        self.study = SafranTotalPrecip(altitude=1800, year_min=1958, year_max=2002, nb_consecutive_days=1)

    def test_durand(self):
        # Test based on Durand paper
        # (some small differences probably due to the fact that SAFRAN model has evolved since then)
        # Test for the mean total precipitation (rainfall + snowfall) between 1958 and 2002
        self.check({
            "Mercantour": 1281,
            'Chablais': 1922,
        })

    def round(self, f):
        return int(f)


class TestSafranTemperature(TestSCMStudy):

    def setUp(self):
        super().setUp()
        self.study = SafranTemperature(altitude=1800, year_min=1958, year_max=2002)

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
