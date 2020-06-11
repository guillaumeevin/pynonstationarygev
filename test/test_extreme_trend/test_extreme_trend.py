import unittest

from projects.exceeding_snow_loads.utils import NON_STATIONARY_TREND_TEST_PAPER_1, NON_STATIONARY_TREND_TEST_PAPER_2


class TestTrendAnalysis(unittest.TestCase):

    def test_nb_parameters_paper1(self):
        trend_test_classes = NON_STATIONARY_TREND_TEST_PAPER_1
        nb_expected = [2, 3, 3, 4, 3, 4, 4, 5]
        for trend_test_class, nb in zip(trend_test_classes, nb_expected):
            self.assertEqual(trend_test_class.total_number_of_parameters_for_unconstrained_model, nb)

    def test_nb_parameters_paper2(self):
        trend_test_classes = NON_STATIONARY_TREND_TEST_PAPER_2
        nb_expected = [2, 3, 3, 4,
                       3, 4, 4, 5,
                       4, 5, 5, 6]
        for trend_test_class, nb in zip(trend_test_classes, nb_expected):
            self.assertEqual(trend_test_class.total_number_of_parameters_for_unconstrained_model, nb)


if __name__ == '__main__':
    unittest.main()
