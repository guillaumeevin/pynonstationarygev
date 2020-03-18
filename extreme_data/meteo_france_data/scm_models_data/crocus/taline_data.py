from collections import OrderedDict

import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth
from extreme_data.meteo_france_data.scm_models_data.scm_constants import ALTITUDES
from root_utils import get_display_name_from_object_type

massif_name = 'Queyras'
study_class = CrocusDepth
all_days = study_class(altitude=0).all_days
altitude_to_queyras_depth = OrderedDict()
for altitude in ALTITUDES[:]:
    study = study_class(altitude=altitude, orientation=90.0)
    if massif_name in study.study_massif_names:
        idx_queyras = study.study_massif_names.index(massif_name)
        queyras_all_daily_series = study.all_daily_series[:, idx_queyras]
        altitude_to_queyras_depth[altitude] = queyras_all_daily_series
df = pd.DataFrame.from_dict(altitude_to_queyras_depth)
df.index = all_days
df.to_csv('{}_{}_90.csv'.format(massif_name, get_display_name_from_object_type(study_class)))