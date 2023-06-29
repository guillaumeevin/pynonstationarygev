import cProfile

from projected_extremes.reviewing.main_quantitative_significance import main_quantitative

if __name__ == '__main__':
    # Example of usage of the profiler
    # main_quantitative()
    cProfile.run('main_quantitative()', sort="cumtime")