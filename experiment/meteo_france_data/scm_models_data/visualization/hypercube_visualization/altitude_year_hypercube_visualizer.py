import numpy as np

from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer


class AltitudeHypercubeVisualizerBis(AltitudeHypercubeVisualizer):

    @staticmethod
    def index_reduction(df, level, **kwargs):
        replace_zero_with_nan = kwargs.get('year_visualization') is not None
        # Take the sum with respect to the years, replace any missing data with np.nan
        if replace_zero_with_nan:
            df = df.sum(axis=1).replace(0.0, np.nan)
        else:
            df = df.sum(axis=1)
        # Take the mean with respect to the level of interest
        return df.mean(level=level)


class Altitude_Hypercube_Year_Visualizer(AltitudeHypercubeVisualizerBis):

    def get_title_plot(self, xlabel, ax_idx=None):
        if ax_idx == self.nb_rows - 1:
            return 'mean starting year'
        return super().get_title_plot(xlabel, ax_idx)

    @property
    def nb_rows(self):
        return super().nb_rows + 1

    def trend_type_reduction_series(self, reduction_function, df_bool):
        series = super().trend_type_reduction_series(reduction_function, df_bool)
        # Create df argmax
        df = df_bool.copy()
        df = (df * df.columns)[df_bool]
        # Reduce and append
        serie = reduction_function(df, year_visualization=True)
        series.append(serie)
        return series
