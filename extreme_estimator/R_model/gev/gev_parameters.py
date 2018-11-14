
class GevParams(object):
    GEV_SCALE = 'scale'
    GEV_LOC = 'loc'
    GEV_SHAPE = 'shape'

    def __init__(self, loc: float, scale: float, shape: float):
        self.location = loc
        self.scale = scale
        self.shape = shape

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**params)

    def to_dict(self) -> dict:
        return {
            self.GEV_LOC: self.location,
            self.GEV_SCALE: self.scale,
            self.GEV_SHAPE: self.shape,
        }

    def rgev(self, nb_obs):
        gev_params = {
            self.GEV_LOC: loc,
            self.GEV_SCALE: scale,
            self.GEV_SHAPE: shape,
        }
        return self.r.rgev(nb_obs, **gev_params)
