import matplotlib.pyplot as plt
import pandas as pd

from extreme_estimator.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import SmoothMarginEstimator
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearAllParametersAllDimsMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.margin_fits.gev.gevmle_fit import GevMleFit
from extreme_estimator.margin_fits.gpd.gpd_params import GpdParams
from extreme_estimator.margin_fits.gpd.gpdmle_fit import GpdMleFit
from experiment.meteo_france_SCM_study.safran.safran import Safran
from extreme_estimator.margin_fits.plot.create_shifted_cmap import get_color_rbga_shifted
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class SafranVisualizer(object):

    def __init__(self, safran: Safran, show=True):
        self.safran = safran
        self.show = show

    @property
    def observations(self):
        return self.safran.observations_annual_maxima

    @property
    def coordinates(self):
        return self.safran.massifs_coordinates

    @property
    def dataset(self):
        return AbstractDataset(self.observations, self.coordinates)

    def fit_and_visualize_estimator(self, estimator):
        estimator.fit()
        axes = estimator.margin_function_fitted.visualize(show=False)
        for ax in axes:
            self.safran.visualize(ax, fill=False, show=False)
        plt.show()

    def visualize_smooth_margin_fit(self):
        # todo: fix some blue points in the corner when we display the margin
        margin_model = LinearAllParametersAllDimsMarginModel(coordinates=self.coordinates)
        estimator = SmoothMarginEstimator(dataset=self.dataset, margin_model=margin_model)
        self.fit_and_visualize_estimator(estimator)

    def visualize_full_fit(self):
        max_stable_model = Smith()
        margin_model = LinearAllParametersAllDimsMarginModel(coordinates=self.coordinates)
        estimator = FullEstimatorInASingleStepWithSmoothMargin(self.dataset, margin_model, max_stable_model)
        self.fit_and_visualize_estimator(estimator)

    def visualize_independent_margin_fits(self, threshold=None, axes=None):
        if threshold is None:
            params_names = GevParams.SUMMARY_NAMES
            df = self.df_gev_mle_each_massif
            # todo: understand how Maurienne could be negative
            # print(df.head())
        else:
            params_names = GpdParams.SUMMARY_NAMES
            df = self.df_gpd_mle_each_massif(threshold)

        if axes is None:
            fig, axes = plt.subplots(1, len(params_names))
            fig.subplots_adjust(hspace=1.0, wspace=1.0)

        for i, gev_param_name in enumerate(params_names):
            ax = axes[i]
            massif_name_to_value = df.loc[gev_param_name, :].to_dict()
            # Compute the middle point of the values for the color map
            values = list(massif_name_to_value.values())
            colors = get_color_rbga_shifted(ax, gev_param_name, values)
            massif_name_to_fill_kwargs = {massif_name: {'color': color} for massif_name, color in
                                          zip(self.safran.safran_massif_names, colors)}
            self.safran.visualize(ax=ax, massif_name_to_fill_kwargs=massif_name_to_fill_kwargs, show=False)

        if self.show:
            plt.show()

    def visualize_cmap(self, massif_name_to_value):
        orig_cmap = plt.cm.coolwarm
        # shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.75, name='shifted')

        massif_name_to_fill_kwargs = {massif_name: {'color': orig_cmap(value)} for massif_name, value in
                                      massif_name_to_value.items()}

        self.safran.visualize(massif_name_to_fill_kwargs=massif_name_to_fill_kwargs)

    """ Statistics methods """

    @property
    def df_gev_mle_each_massif(self):
        # Fit a margin_fits on each massif
        massif_to_gev_mle = {massif_name: GevMleFit(self.safran.observations_annual_maxima.loc[massif_name]).gev_params.summary_serie
                             for massif_name in self.safran.safran_massif_names}
        return pd.DataFrame(massif_to_gev_mle, columns=self.safran.safran_massif_names)

    def df_gpd_mle_each_massif(self, threshold):
        # Fit a margin fit on each massif
        massif_to_gev_mle = {massif_name: GpdMleFit(self.safran.df_all_snowfall_concatenated[massif_name],
                                                    threshold=threshold).gpd_params.summary_serie
                             for massif_name in self.safran.safran_massif_names}
        return pd.DataFrame(massif_to_gev_mle, columns=self.safran.safran_massif_names)
