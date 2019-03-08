import os.path as op
import unittest

import pandas as pd

from experiment.meteo_france_SCM_study.crocus.crocus import ExtendedCrocusSwe
from experiment.meteo_france_SCM_study.visualization.study_visualization.main_study_visualizer import study_iterator
from experiment.meteo_france_SCM_study.safran.safran import SafranSnowfall, ExtendedSafranSnowfall, SafranTemperature, \
    SafranPrecipitation
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from test.test_utils import load_scm_studies


class TestSCMAllStudy(unittest.TestCase):

    def test_extended_run(self):
        for study_class in [ExtendedSafranSnowfall, ExtendedCrocusSwe]:
            for study in study_iterator(study_class, only_first_one=True, both_altitude=False, verbose=False):
                study_visualizer = StudyVisualizer(study, show=False, save_to_file=False)
                study_visualizer.visualize_all_mean_and_max_graphs()
        self.assertTrue(True)

    def test_scm_daily_data(self):
        for study in load_scm_studies():
            time_serie = study.year_to_daily_time_serie_array[1958]
            self.assertTrue(time_serie.ndim == 2, msg='for {} ndim={}'.format(study.__repr__(), time_serie.ndim))
            self.assertTrue(time_serie.shape[1] in [21, 23])
            self.assertTrue(len(time_serie) in [365, 366],
                            msg="current time serie length for {} is {}".format(study.__repr__(), len(time_serie)))


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
        assert not set(self.study.safran_massif_names).symmetric_difference(set(df_centroid['NOM']))


class TestSCMPrecipitation(TestSCMStudy):

    def setUp(self) -> None:
        super().setUp()
        self.study = SafranPrecipitation(altitude=1800, year_min=1958, year_max=2002)

    # def test_durand(self):
    #     # Test based on Durand paper
    #     # Test for the mean temperature between 1958 and 2002
    #     self.check({
    #         "Mercantour": 1340,
    #         'Chablais': 1928,
    #     })

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
            "Mercantour": 5.1,
            'Chablais': 3.4,
        })

    def round(self, f):
        return round(f, 1)


if __name__ == '__main__':
    unittest.main()
