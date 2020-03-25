from datetime import datetime

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.utils import date_to_str
from root_utils import get_full_path

# Type of Weather

ANTICYCLONIC = 'Anticyclonic'

CENTRAL_DEPRESSION = 'Central Depression'

EAST_RETURN = 'East Return'

NORTHEAST_CIRCULATION = 'Northeast Circulation'

SOUTH_CIRCULATION = 'South Circulation'

SOUTHWEST_CIRCULATION = 'Southwest Circulation'

STEADY_OCEANIC = 'Steady Oceanic'

ATLANTIC_WAVE = 'Atlantic Wave'

wp_int_to_wp_str = {
    1: ATLANTIC_WAVE,
    2: STEADY_OCEANIC,
    3: SOUTHWEST_CIRCULATION,
    4: SOUTH_CIRCULATION,
    5: NORTHEAST_CIRCULATION,
    6: EAST_RETURN,
    7: CENTRAL_DEPRESSION,
    8: ANTICYCLONIC,
}


def load_df_weather_types() -> pd.DataFrame:
    relative_path = r'local/EDF_data/Weather_types/CatalogueTT_EDF_France0_5308.txt'
    edf_filepath = get_full_path(relative_path=relative_path)
    global df
    weather_types = []
    with open(edf_filepath, 'rb') as f:
        for i, l in enumerate(f):
            if i >= 7:
                l = str(l).split('"')[1:]
                wp_int = int(l[1][2])
                wp_str = wp_int_to_wp_str[wp_int]
                day, month, year = [int(e) for e in l[0].split('/')]
                date_str = date_to_str(datetime(year=year, month=month, day=day))
                weather_types.append((date_str, wp_str))
    df = pd.DataFrame(weather_types, columns=['Date', 'WP'])
    df.set_index('Date', inplace=True)
    return df
