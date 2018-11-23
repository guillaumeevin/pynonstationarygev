from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates
from spatio_temporal_dataset.coordinates.unidimensional_coordinates.coordinates_1D import UniformCoordinates

COORDINATES = [UniformCoordinates, CircleCoordinates, AlpsStation3DCoordinatesWithAnisotropy]


def load_coordinates(nb_points):
    return [coordinate_class.from_nb_points(nb_points=nb_points) for coordinate_class in COORDINATES]
