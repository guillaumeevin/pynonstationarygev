import unittest

from test.test_utils import load_safran_objects


class TestFullEstimators(unittest.TestCase):

    def test_gev_mle_per_massif(self):
        safran_1800_one_day = load_safran_objects()[0]
        df = safran_1800_one_day.df_gev_mle_each_massif
        self.assertAlmostEqual(df.values.sum(), 1131.4551665871832)


if __name__ == '__main__':
    unittest.main()
