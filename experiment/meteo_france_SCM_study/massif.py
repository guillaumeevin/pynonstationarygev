from utils import first

MASSIF_NAMES_1800 = ['Chablais', 'Aravis', 'Mont-Blanc', 'Bauges', 'Beaufortain',
                     'Haute-Tarentaise', 'Chartreuse', 'Belledonne', 'Maurienne', 'Vanoise',
                     'Haute-Maurienne', 'Grandes-Rousses', 'Thabor', 'Vercors', 'Oisans',
                     'Pelvoux', 'Queyras', 'Devoluy', 'Champsaur', 'Parpaillon', 'Ubaye',
                     'Alpes-Azur', 'Mercantour']
# Some massif like Chartreuse do not have massif whose altitude is higher or equal to 2400
MASSIF_NAMES_2400 = ['Chablais', 'Aravis', 'Mont-Blanc', 'Beaufortain', 'Haute-Tarentaise',
                     'Belledonne', 'Maurienne', 'Vanoise', 'Haute-Maurienne',
                     'Grandes-Rousses', 'Thabor', 'Vercors', 'Oisans', 'Pelvoux', 'Queyras',
                     'Devoluy', 'Champsaur', 'Parpaillon', 'Ubaye', 'Alpes-Azur',
                     'Mercantour']


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


def safran_massif_names_from_datasets(datasets, altitude):
    # Massifs names are extracted from SAFRAN dataset
    reference_massif_list = MASSIF_NAMES_1800 if altitude == 1800 else MASSIF_NAMES_2400
    if hasattr(datasets[0], 'massifsList'):
        # Assert the all the datasets have the same indexing for the massif
        assert len(set([dataset.massifsList for dataset in datasets])) == 1
        # List of the name of the massif used by all the SAFRAN datasets
        massifs = [Massif.from_str(massif_str) for massif_str in first(datasets).massifsList.split('/')]
        # IMPORTANT: Sort the massif names
        massifs = sorted(massifs, key=lambda massif: massif.id)
        safran_names = [massif.name for massif in massifs]
        assert reference_massif_list == safran_names, '{} \n{}'.format(reference_massif_list, safran_names)
    return reference_massif_list
