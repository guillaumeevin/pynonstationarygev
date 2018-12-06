from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


# class TemporalCoordinates(AbstractCoordinates):
#     pass
#
#
#     @classmethod
#     def from_nb_points(cls, nb_points, train_split_ratio: float = None, start=-1.0, end=1.0):
#         # Sample uniformly inside the circle
#         df = cls.df_spatial(nb_points, start, end)
#         return cls.from_df(df, train_split_ratio)
#
#     @classmethod
#     def df_spatial(cls, nb_points, start=-1.0, end=1.0):
#         axis_coordinates = np.array(r.runif(nb_points, min=start, max=end))
#         df = pd.DataFrame.from_dict({cls.COORDINATE_X: axis_coordinates})
#         return df
#
#     @classmethod
#     def from_nb_points(cls, nb_points, train_split_ratio: float = None, nb_time_steps=1, max_radius=1.0):
#         assert isinstance(nb_time_steps, int) and nb_time_steps >= 1
#         df_spatial = UniformSpatialCoordinates.df_spatial(nb_points)
#         df_time_steps = []
#         for t in range(nb_time_steps):
#             df_time_step = df_spatial.copy()
#             df_time_step[cls.COORDINATE_T] = t
#             df_time_steps.append(df_time_step)
#         df_time_steps = pd.concat(df_time_steps, ignore_index=True)
#         print(df_time_steps)
#         return cls.from_df(df=df_time_steps, train_split_ratio=train_split_ratio, slicer_class=SpatioTemporalSlicer)