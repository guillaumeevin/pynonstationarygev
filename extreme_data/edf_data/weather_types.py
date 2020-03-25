from datetime import datetime

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.utils import date_to_str
from root_utils import get_full_path

relative_path = r'local/EDF_data/Weather_types/CatalogueTT_EDF_France0_5308.txt'
edf_filepath = get_full_path(relative_path=relative_path)


def load_df_weather_types() -> pd.DataFrame:
    global df
    weather_types = []
    with open(edf_filepath, 'rb') as f:
        for i, l in enumerate(f):
            if i >= 7:
                l = str(l).split('"')[1:]
                wp = int(l[1][2])
                day, month, year = [int(e) for e in l[0].split('/')]
                date_str = date_to_str(datetime(year=year, month=month, day=day))
                weather_types.append((date_str, wp))
    df = pd.DataFrame(weather_types, columns=['Date', 'WP'])
    df.set_index('Date', inplace=True)
    return df


