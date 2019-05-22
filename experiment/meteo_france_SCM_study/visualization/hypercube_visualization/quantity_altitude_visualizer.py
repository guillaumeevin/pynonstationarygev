import pandas as pd

from experiment.meteo_france_SCM_study.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer


class QuantityAltitudeHypercubeVisualizer(AltitudeHypercubeVisualizer):

    @property
    def study_title(self):
        return 'Quantity Altitude Study'

    def subtitle_to_reduction_function(self, reduction_function, level=None, add_detailed_plot=False, subtitle=None):
        subtitle_to_reduction_function = super().subtitle_to_reduction_function(reduction_function,
                                                                                level, add_detailed_plot,
                                                                                'global')

        def get_function_from_tuple(tuple_for_axis_0):
            def f(df_bool: pd.DataFrame):
                # Loc with a tuple with respect the axis 0
                df_bool = df_bool.loc[tuple_for_axis_0, :].copy()
                # Apply the reduction function
                return reduction_function(df_bool) if level is None else reduction_function(df_bool, level-1)
            return f

        # Add the detailed plot, taken by loc with respect to the first index
        if add_detailed_plot:
            tuples_axis_0 = self.tuple_values(idx=0)
            for tuple_axis_0 in tuples_axis_0:
                subtitle_to_reduction_function[tuple_axis_0] = get_function_from_tuple(tuple_axis_0)
        return subtitle_to_reduction_function

    @property
    def quantities(self):
        return self.tuple_values(idx=0)

    @property
    def altitudes(self):
        return self.tuple_values(idx=1)

    @property
    def altitude_index_level(self):
        return 1

    @property
    def massif_index_level(self):
        return 2
