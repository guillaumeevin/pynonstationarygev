import unittest

from extreme_estimator.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.extreme_models.margin_model.linear_margin_model.linear_margin_model import ConstantMarginModel, \
    LinearMarginModelExample
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import CovarianceFunction
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Schlather
from extreme_estimator.extreme_models.utils import r
from test.test_unitary.test_rmaxstab.test_rmaxstab_with_margin import TestRMaxStabWithMarginConstant
from test.test_unitary.test_unitary_abstract import TestUnitaryAbstract


class TestMaxStableFitWithConstantMargin(TestUnitaryAbstract):

    @property
    def r_output(self):
        TestRMaxStabWithMarginConstant.r_code()
        r("""
        shape.form = shape ~ 1
        loc.form = loc ~ 1
        scale.form = scale ~ 1
        res = fitmaxstab(data, locations, "whitmat", loc.form, scale.form, shape.form)
        """)
        return self.r_fitted_values_from_res_variable

    @property
    def python_output(self):
        dataset = TestRMaxStabWithMarginConstant.python_code()
        max_stable_model = Schlather(covariance_function=CovarianceFunction.whitmat, use_start_value=False)
        margin_model = ConstantMarginModel(dataset.coordinates)
        full_estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset, margin_model,
                                                                    max_stable_model)
        full_estimator.fit()
        return full_estimator.result_from_fit.all_parameters

    def test_max_stable_fit_with_constant_margin(self):
        self.compare()


class TestMaxStableFitWithLinearMargin(TestUnitaryAbstract):

    @property
    def r_output(self):
        TestRMaxStabWithMarginConstant.r_code()
        r("""
        loc.form <- loc ~ lat
        scale.form <- scale ~ lon
        shape.form <- shape ~ lon
        res = fitmaxstab(data, locations, "whitmat", loc.form, scale.form, shape.form)
        """)
        return self.r_fitted_values_from_res_variable

    @property
    def python_output(self):
        dataset = TestRMaxStabWithMarginConstant.python_code()
        max_stable_model = Schlather(covariance_function=CovarianceFunction.whitmat, use_start_value=False)
        margin_model = LinearMarginModelExample(dataset.coordinates)
        full_estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset, margin_model,
                                                                    max_stable_model)
        full_estimator.fit()
        return full_estimator.result_from_fit.all_parameters

    def test_max_stable_fit_with_linear_margin(self):
        self.compare()


# class TestMaxStableFitWithSpline(TestUnitaryAbstract):
#
#     @property
#     def r_output(self):
#         TestRMaxStabWithMarginConstant.r_code()
#         r("""
#         # Code inspired from Johan code
#         n.knots_x = 1
#         n.knots_y = 2
#         knots = quantile(locations[,1], prob=1:n.knots_x/(n.knots_x+1))
#         knots2 = quantile(locations[,2], prob=1:n.knots_y/(n.knots_y+1))
#         knots_tot = cbind(knots,knots2)
#
#         loc.form <- y ~ rb(locations[,1], knots = knots, degree = 3, penalty = .5)
#         scale.form <- y ~ rb(locations[,2], knots = knots2, degree = 3, penalty = .5)
#         shape.form <- y ~ rb(locations, knots = knots_tot, degree = 3, penalty = .5)
#         """)
#         return self.r_fitted_values_from_res_variable
#
#     @property
#     def python_output(self):
#         dataset = TestRMaxStabWithMarginConstant.python_code()
#         max_stable_model = Schlather(covariance_function=CovarianceFunction.whitmat, use_start_value=False)
#         margin_model = LinearMarginModelExample(dataset.coordinates)
#         full_estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset, margin_model,
#                                                                     max_stable_model)
#         full_estimator.fit()
#         return full_estimator.fitted_values
#
#     def test_max_stable_fit_with_spline_margin(self):
#         self.compare()


if __name__ == '__main__':
    unittest.main()
