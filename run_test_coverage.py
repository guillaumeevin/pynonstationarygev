import os.path as op
from root_utils import terminal_command, get_root_path

"""
To run all my test do:  python -m unittest

Test coverage is only in the professional version of pycharm.
To configure the extent of my coverage: modify the .coveragerc
To run test coverage in the terminal: coverage run -m unittest
Then a hidden file .coverage is create at the root
To create a html file from it do: coverage html
Then open the htlmcov/index.html in your browser
"""


def run_test_coverage():
    if op.exists(op.join(get_root_path(), "coverage_html_report")):
        terminal_command("rm -r coverage_html_report")
    terminal_command("coverage run -m unittest")


def display_last_test_coverage():
    terminal_command("coverage html")
    terminal_command("firefox coverage_html_report/index.html &")


if __name__ == '__main__':
    # run_test_coverage()
    display_last_test_coverage()
