import numpy as np

from experiment.meteo_france_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer


class Altitude_Hypercube_Year_Visualizer(AltitudeHypercubeVisualizer):

    def get_title_plot(self, xlabel, ax_idx=None):
        if ax_idx == self.nb_axes - 1:
            return 'mean starting year'
        return super().get_title_plot(xlabel, ax_idx)

    @property
    def nb_axes(self):
        return super().nb_axes + 1

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

    def trend_type_reduction(self, reduction_function, display_trend_type):
        series, df_bool = super().trend_type_reduction(reduction_function, display_trend_type)
        # Create df argmax
        df = df_bool.copy()
        df = (df * df.columns)[df_bool]
        # Reduce and append
        serie = reduction_function(df, year_visualization=True)
        series.append(serie)
        return series, df_bool
