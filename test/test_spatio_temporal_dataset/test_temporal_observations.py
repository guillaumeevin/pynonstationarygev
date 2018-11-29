import unittest
import numpy as np

import pandas as pd

from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import AbstractSpatioTemporalObservations


class TestTemporalObservations(unittest.TestCase):
    DISPLAY = False

    def test_set_maxima_gev(self):
        df = pd.DataFrame.from_dict({'ok': [2, 5]})
        temporal_observation = AbstractSpatioTemporalObservations(df_maxima_frech=df)
        example = np.array([[3], [6]])
        temporal_observation.set_maxima_frech(maxima_frech_values=example)
        maxima_frech = temporal_observation.maxima_frech()
        self.assertTrue(np.equal(example, maxima_frech).all(), msg="{} {}".format(example, maxima_frech))


if __name__ == '__main__':
    unittest.main()
