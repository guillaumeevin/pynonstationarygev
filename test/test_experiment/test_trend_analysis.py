import unittest

from papers.exceeding_snow_loads.paper_utils import NON_STATIONARY_TREND_TEST_PAPER


class TestTrendAnalysis(unittest.TestCase):

    def test_nb_parameters(self):
        trend_test_classes = NON_STATIONARY_TREND_TEST_PAPER
        nb_expected = [2, 3, 3, 4, 3, 4, 4, 5]
        for trend_test_class, nb in zip(trend_test_classes, nb_expected):
            self.assertEqual(trend_test_class.total_number_of_parameters_for_unconstrained_model, nb)


if __name__ == '__main__':
    unittest.main()
