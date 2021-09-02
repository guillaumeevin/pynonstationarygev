from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.spline_margin_model.spline_margin_model import SplineMarginModel


class NonStationaryTwoLinearLocationModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        # Degree 1, Two Linear sections for the location
        return super().load_margin_function({GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)]})


class NonStationaryTwoLinearScaleModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        # Degree 1, Two Linear sections for the scale parameters
        return super().load_margin_function({GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)]})


class NonStationaryTwoLinearShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        # Degree 1, Two Linear sections for the shape parameters
        return super().load_margin_function({GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)]})


# Two Linearity with one linearity

class NonStationaryTwoLinearLocationOneLinearScaleModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


class NonStationaryTwoLinearLocationOneLinearShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


class NonStationaryTwoLinearScaleOneLinearLocModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


class NonStationaryTwoLinearScaleOneLinearShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


class NonStationaryTwoLinearShapeOneLinearLocModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


class NonStationaryTwoLinearShapeOneLinearScaleModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


# Two Linearity with two one linearity

class NonStationaryTwoLinearLocationOneLinearScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


class NonStationaryTwoLinearScaleOneLinearLocAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


class NonStationaryTwoLinearShapeOneLinearLocAndScaleModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1)]
        })


# two linearity two times

class NonStationaryTwoLinearLocationAndScaleModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
        })


class NonStationaryTwoLinearScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
        })


class NonStationaryTwoLinearLocationAndShape(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
        })


# two linearity two times and one linearity

class NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1)],
        })


class NonStationaryTwoLinearScaleAndShapeOneLinearLocModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1)],
        })


class NonStationaryTwoLinearLocationAndShapeOneLinearScaleModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1)],
        })


# two linearity three times

class NonStationaryTwoLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)],
        })


# three linearity three times

class NonStationaryThreeLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 3)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 3)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 3)],
        })


# four linearity three times

class NonStationaryFourLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 4)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 4)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 4)],
        })


class NonStationaryFiveLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 5)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 5)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 5)],
        })


class NonStationarySixLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 6)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 6)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 6)],
        })


class NonStationarySevenLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 7)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 7)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 7)],
        })


class NonStationaryEightLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 8)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 8)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 8)],
        })


class NonStationaryNineLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 9)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 9)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 9)],
        })


class NonStationaryTenLinearLocationAndScaleAndShapeModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 10)],
            GevParams.SHAPE: [(self.coordinates.idx_temporal_coordinates, 1, 10)],
            GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 10)],
        })
