import unittest

from numpy.random.mtrand import gumbel

from extreme_fit.distribution.gumbel.gumbel_gof import \
    cramer_von_mises_and_anderson_darling_tests_pvalues_for_gumbel_distribution
from extreme_fit.model.utils import set_seed_for_test


class TestGumbel(unittest.TestCase):

    def test_gof_tests(self):
        set_seed_for_test(seed=42)
        data = gumbel(size=60)
        res = cramer_von_mises_and_anderson_darling_tests_pvalues_for_gumbel_distribution(data)
        cramer_pvalue, anderson_pvalue = res
        self.assertGreater(cramer_pvalue, 0.05)
        self.assertGreater(anderson_pvalue, 0.05)


if __name__ == '__main__':
    unittest.main()
