from root_utils import classproperty


class AbstractCluster(object):

    @classproperty
    def massif_name_to_cluster_id(cls):
        raise NotImplemented

    @classproperty
    def cluster_id_to_cluster_name(cls):
        raise NotImplemented

    @classproperty
    def cluster_id_to_massif_names(cls):
        cluster_id_to_massif_names = {}
        for cluster_id in cls.cluster_ids:
            massif_names = []
            for massif_name, cluster_id2 in cls.massif_name_to_cluster_id.items():
                if cluster_id == cluster_id2:
                    massif_names.append(massif_name)
            cluster_id_to_massif_names[cluster_id] = massif_names
        return cluster_id_to_massif_names

    @classproperty
    def cluster_ids(cls):
        return list(cls.cluster_id_to_cluster_name.keys())