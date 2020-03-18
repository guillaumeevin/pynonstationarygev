from typing import Dict

import pandas as pd

from extreme_data.eurocode_data.utils import EUROCODE_ALTITUDES
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends, ModelSubsetForUncertainty


def uncertainty_interval_size(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """ Plot one graph for each non-stationary context
    :return:
    """
    altitude_to_visualizer = {a: v for a, v in altitude_to_visualizer.items() if a in EUROCODE_ALTITUDES}
    visualizer = list(altitude_to_visualizer.values())[0]
    for a, v in altitude_to_visualizer.items():
        print(a)
        interval_size(v)


def interval_size(v: StudyVisualizerForNonStationaryTrends):
    d = v.all_massif_name_to_eurocode_uncertainty_for_minimized_aic_model_class(
        model_subset_for_uncertainty=ModelSubsetForUncertainty.stationary_gev)
    # what we want is the confidence interval for the shape parameter
    d = {m: [e.confidence_interval[0], e.confidence_interval[1], e.confidence_interval[1] - e.confidence_interval[0]]
         for m, e in d.items()}
    df = pd.DataFrame(d).transpose()
    print((df.head()))
    print(df.describe())
