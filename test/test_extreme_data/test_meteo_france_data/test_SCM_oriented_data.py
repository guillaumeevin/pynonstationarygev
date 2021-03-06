import os.path as op
import time
import unittest

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSwe3Days
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from extreme_data.meteo_france_data.scm_models_data.utils import SLOPES


class TestSCMOrientedData(unittest.TestCase):

    def test_various_orientations(self):
        for altitude in [900, 1800]:
            for slope in SLOPES:
                for orientation in [None, 45.0, 180.0][:2]:
                    for study_class in [SafranSnowfall, CrocusSwe3Days][:]:
                        study = study_class(altitude=altitude, orientation=orientation, slope=slope, year_max=1959, multiprocessing=False)
                        assert study.year_to_daily_time_serie_array[1959].shape == (365, 23)


if __name__ == '__main__':
    unittest.main()
