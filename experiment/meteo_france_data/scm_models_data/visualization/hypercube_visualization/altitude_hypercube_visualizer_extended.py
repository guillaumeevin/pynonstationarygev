from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.abstract_hypercube_visualizer import \
    AbstractHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer


class AltitudeHypercubeVisualizerExtended(AltitudeHypercubeVisualizer):

    def df_bool(self, display_trend_type):
        df_bool = super().df_bool(display_trend_type)
        print(df_bool)
        return df_bool


class AltitudeYearHypercubeVisualizerExtended(AltitudeHypercubeVisualizerExtended, Altitude_Hypercube_Year_Visualizer):
    pass