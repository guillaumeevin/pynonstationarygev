import pandas as pd

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
                weather_types.append((l[0], int(l[1][2])))
    df = pd.DataFrame(weather_types, columns=['Date', 'WP'])
    df.set_index('Date', inplace=True)
    return df


