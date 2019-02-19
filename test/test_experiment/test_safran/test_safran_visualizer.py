# import unittest
#
# from experiment.meteo_france_SCM_study.safran import Safran
# from experiment.meteo_france_SCM_study.safran_visualizer import SafranVisualizer
#
#
# class TestSafranVisualizer(unittest.TestCase):
#     DISPLAY = False
#
#     def setUp(self):
#         super().setUp()
#         self.safran = Safran(1800, 1)
#         self.safran_visualizer = SafranVisualizer(self.safran, show=self.DISPLAY)
#
#     def tearDown(self) -> None:
#         self.assertTrue(True)
#
#     def test_safran_smooth_margin_estimator(self):
#         self.safran_visualizer.visualize_smooth_margin_fit()
#
#     def test_safran_independent_margin_fits(self):
#         self.safran_visualizer.visualize_independent_margin_fits()
#
#     def test_safran_full_estimator(self):
#         self.safran_visualizer.visualize_full_fit()
#
#
# if __name__ == '__main__':
#     unittest.main()
