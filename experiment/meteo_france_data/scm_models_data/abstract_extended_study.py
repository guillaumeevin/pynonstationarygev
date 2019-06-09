import numpy as np
from collections import OrderedDict

from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from utils import classproperty


class AbstractExtendedStudy(AbstractStudy):



    @classproperty
    def region_names(cls):
        return ['Alps', 'Northern Alps', 'Central Alps', 'Southern Alps', 'Extreme South Alps']

    @property
    def nb_region_names(self):
        return len(self.region_names)

    @property
    def study_massif_names(self):
        return self.region_names + super().study_massif_names

    @classproperty
    def massif_name_to_region_name(cls):
        df_centroid = cls.load_df_centroid()
        return OrderedDict(zip(df_centroid.index, df_centroid['REGION']))

    @classproperty
    def region_name_to_massif_names(cls):
        return {k: [cls.original_safran_massif_id_to_massif_name[i] for i in v]
                for k, v in cls.region_name_to_massif_ids.items()}

    @classproperty
    def region_name_to_massif_ids(cls):
        region_name_to_massifs_ids = {}
        for region_name_loop in cls.region_names:
            # We use "is in" so that the "Alps" will automatically regroup all the massif data
            massif_names_belong_to_the_group = [massif_name
                                                for massif_name, region_name in cls.massif_name_to_region_name.items()
                                                if region_name_loop in region_name]
            massif_ids_belong_to_the_group = [massif_id
                                              for massif_id, massif_name in
                                              cls.original_safran_massif_id_to_massif_name.items()
                                              if massif_name in massif_names_belong_to_the_group]
            region_name_to_massifs_ids[region_name_loop] = massif_ids_belong_to_the_group
        return region_name_to_massifs_ids

    """ Properties """

    def massifs_coordinates_for_display(self) -> AbstractSpatialCoordinates:
        raise ValueError('Coordinates are meaningless for an extended study')

    @property
    def _year_to_daily_time_serie_array(self) -> OrderedDict:
        return self._year_to_extended_time_serie(aggregation_function=np.mean)

    @property
    def _year_to_max_daily_time_serie(self):
        return self._year_to_extended_time_serie(aggregation_function=np.max)

    def _year_to_extended_time_serie(self, aggregation_function) -> OrderedDict:
        year_to_extended_time_serie = OrderedDict()
        for year, old_time_serie in super()._year_to_daily_time_serie_array.items():
            new_time_serie = np.zeros([len(old_time_serie), len(self.study_massif_names)])
            new_time_serie[:, self.nb_region_names:] = old_time_serie
            for i, region_name in enumerate(self.region_names):
                massifs_ids_belong_to_region = self.region_name_to_massif_ids[region_name]
                aggregated_time_serie = aggregation_function(old_time_serie[:, massifs_ids_belong_to_region], axis=1)
                new_time_serie[:, i] = aggregated_time_serie
            year_to_extended_time_serie[year] = new_time_serie
        return year_to_extended_time_serie
