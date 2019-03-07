import os.path as op
import unittest

import pandas as pd

from experiment.meteo_france_SCM_study.crocus.crocus import ExtendedCrocusSwe
from experiment.meteo_france_SCM_study.visualization.study_visualization.main_study_visualizer import study_iterator
from experiment.meteo_france_SCM_study.safran.safran import SafranSnowfall, ExtendedSafranSnowfall
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from test.test_utils import load_scm_studies


class TestSCMStudy(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.study = SafranSnowfall()

    def test_massif_safran(self):
        df_centroid = pd.read_csv(op.join(self.study.map_full_path, 'coordonnees_massifs_alpes.csv'))
        # Assert that the massif names are the same between SAFRAN and the coordinate file
        assert not set(self.study.safran_massif_names).symmetric_difference(set(df_centroid['NOM']))

    def test_extended_run(self):
        for study_class in [ExtendedSafranSnowfall, ExtendedCrocusSwe]:
            for study in study_iterator(study_class, only_first_one=True, both_altitude=False, verbose=False):
                study_visualizer = StudyVisualizer(study, show=False, save_to_file=False)
                study_visualizer.visualize_all_mean_and_max_graphs()
        self.assertTrue(True)

    def test_scm_daily_data(self):
        for study in load_scm_studies():
            time_serie = study.year_to_daily_time_serie[1958]
            self.assertTrue(time_serie.ndim == 2, msg='for {} ndim={}'.format(study.__repr__(), time_serie.ndim))
            self.assertTrue(len(time_serie) in [365, 366],
                            msg="current time serie length for {} is {}".format(study.__repr__(), len(time_serie)))


if __name__ == '__main__':
    unittest.main()
