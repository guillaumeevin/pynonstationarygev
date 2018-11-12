import unittest
import pandas as pd

class TestPipeline(unittest.TestCase):

    def main_pipeline(self):
        # Select a type of marginals (either spatial, spatio temporal, temporal)
        #  this will define the dimension of the climatic space of interest
        pass
        # Select the max stable

        #  Define an optimization process
        # The algo: In 1 time, in 2 times, ..., or more complex patterns
        # This algo have at least main procedures (that might be repeated several times)

        # For each procedure, we shall define:
        # - The loss
        # - The optimization method for each part of the process

    def blanchet_smooth_pipeline(self):
        pass
        # Spatial marginal

        # NO MAX STABLE

        # Procedure:
        # Optimization of a single likelihood process that sums up the likelihood of all the terms.

    def padoan_extreme_pipeline(self):
        pass
        # Spatial marginal

        # todo: question, when we are optimizing the full Pairwise loss, are we just optimization the relations ?
        # or ideally do we need to add the term of order 1

    def gaume(self):
        # Combining the 2
        pass




    def test_pipeline_spatial(self):
        pass
        # Sample from a

        # Fit the spatio temporal experiment margin

        # Fit the max stable process

if __name__ == '__main__':
    unittest.main()