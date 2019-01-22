from utils import first


class Massif(object):

    def __init__(self, name: str, id: int, lat: float, lon: float) -> None:
        self.lon = lon
        self.lat = lat
        self.id = id
        self.name = name

    @classmethod
    def from_str(cls, s: str):
        name, id, lat, lon = s.split(',')
        return cls(name.strip(), int(id), float(lat), float(lon))


def safran_massif_names_from_datasets(datasets):
    # Assert the all the datasets have the same indexing for the massif
    assert len(set([dataset.massifsList for dataset in datasets])) == 1
    # List of the name of the massif used by all the SAFRAN datasets
    return [Massif.from_str(massif_str).name for massif_str in first(datasets).massifsList.split('/')]