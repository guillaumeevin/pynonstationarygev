import os.path as op
import unittest

import pandas as pd

from experiment.meteo_france_SCM_study.crocus.crocus import ExtendedCrocusSwe
from experiment.meteo_france_SCM_study.main_visualize import study_iterator
from experiment.meteo_france_SCM_study.safran.safran import Safran, ExtendedSafran
from experiment.meteo_france_SCM_study.safran.safran_visualizer import StudyVisualizer


class TestSCMStudy(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.study = Safran()

    def test_massif_safran(self):
        df_centroid = pd.read_csv(op.join(self.study.map_full_path, 'coordonnees_massifs_alpes.csv'))
        # Assert that the massif names are the same between SAFRAN and the coordinate file
        assert not set(self.study.safran_massif_names).symmetric_difference(set(df_centroid['NOM']))

    def test_extended_run(self):
        for study_class in [ExtendedSafran, ExtendedCrocusSwe]:
            for study in study_iterator(study_class, only_first_one=True, both_altitude=True, verbose=False):
                study_visualizer = StudyVisualizer(study, show=False, save_to_file=False)
                study_visualizer.visualize_all_kde_graphs()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
