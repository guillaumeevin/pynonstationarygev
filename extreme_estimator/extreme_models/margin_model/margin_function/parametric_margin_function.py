from typing import Dict

from extreme_estimator.extreme_models.margin_model.margin_function.independent_margin_function import \
    IndependentMarginFunction


class ParametricMarginFunction(IndependentMarginFunction):

    @property
    def form_dict(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def coef_dict(self) -> Dict[str, float]:
        raise NotImplementedError