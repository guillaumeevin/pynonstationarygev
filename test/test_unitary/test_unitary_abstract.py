import unittest

from extreme_fit.model.utils import set_seed_r, r


class TestUnitaryAbstract(unittest.TestCase):

    def setUp(self):
        r("""
        n.site <- 30
        locations <- matrix(runif(2*n.site, 0, 10), ncol = 2)
        colnames(locations) <- c("lon", "lat")
        """)

    @property
    def r_fitted_values_from_res_variable(self):
        res = r.res
        fitted_values = res.rx2('fitted.values')
        fitted_values = {key: fitted_values.rx2(key)[0] for key in fitted_values.names}
        return fitted_values

    @property
    def r_output(self):
        pass

    @property
    def python_output(self):
        pass

    def compare(self):
        set_seed_r()
        r_output = self.r_output
        set_seed_r()
        python_output = self.python_output
        self.assertEqual(r_output, python_output, msg="python: {}, r: {}".format(python_output, r_output))


