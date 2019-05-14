
"""
ZS was extracted from a netcdf file
"""
import numpy as np

ZS = """[ 300.  600.  900. 1200. 1500. 1800. 2100. 2400. 2700. 3000. 3300.  300.
600.  900. 1200. 1500. 1800. 2100. 2400. 2700. 3000.  300.  600.  900.
1200. 1500. 1800. 2100. 2400. 2700. 3000. 3300. 3600. 3900. 4200. 4500.
4800.    0.  300.  600.  900. 1200. 1500. 1800. 2100. 2400.  300.  600.
900. 1200. 1500. 1800. 2100. 2400. 2700. 3000.  600.  900. 1200. 1500.
1800. 2100. 2400. 2700. 3000. 3300. 3600. 3900.    0.  300.  600.  900.
1200. 1500. 1800. 2100.    0.  300.  600.  900. 1200. 1500. 1800. 2100.
2400. 2700. 3000.    0.  300.  600.  900. 1200. 1500. 1800. 2100. 2400.
2700. 3000. 3300.  300.  600.  900. 1200. 1500. 1800. 2100. 2400. 2700.
3000. 3300. 3600. 3900.  900. 1200. 1500. 1800. 2100. 2400. 2700. 3000.
3300. 3600. 3900.  600.  900. 1200. 1500. 1800. 2100. 2400. 2700. 3000.
3300. 3600.  900. 1200. 1500. 1800. 2100. 2400. 2700. 3000. 3300.    0.
300.  600.  900. 1200. 1500. 1800. 2100. 2400.    0.  300.  600.  900.
1200. 1500. 1800. 2100. 2400. 2700. 3000. 3300. 3600. 3900. 4200.  900.
1200. 1500. 1800. 2100. 2400. 2700. 3000. 3300. 3600. 3900. 4200.  900.
1200. 1500. 1800. 2100. 2400. 2700. 3000. 3300.  300.  600.  900. 1200.
1500. 1800. 2100. 2400. 2700. 3000.  600.  900. 1200. 1500. 1800. 2100.
2400. 2700. 3000. 3300. 3600.  600.  900. 1200. 1500. 1800. 2100. 2400.
2700. 3000. 3300.  600.  900. 1200. 1500. 1800. 2100. 2400. 2700. 3000.
3300. 3600.    0.  300.  600.  900. 1200. 1500. 1800. 2100. 2400. 2700.
3000. 3300.  300.  600.  900. 1200. 1500. 1800. 2100. 2400. 2700. 3000.
3300.  300.  600.  900. 1200. 1500. 1800. 2100. 2400. 2700. 3000.]"""

ZS_INT = [int(float(e)) for e in ZS[1:-1].split()]
ALTITUDES = sorted(set(ZS_INT))


# Create a ZS_INT with only the 23 first massifs
ZS_INT_23 = ZS_INT[:-10].copy()

# Create a ZS_INT with np.nan all altitudes corresponding to the 24th massif
ZS_INT_MASK = np.array(ZS_INT)
ZS_INT_MASK[-10:] = np.nan

# Longitudes and Latitudes in degrees
LONGITUDES = [6.64493, 6.64493, 6.64493, 6.64493, 6.64493, 6.64493, 6.64493, 6.64493, 6.64493, 6.64493, 6.64493, 6.39738, 6.39738, 6.39738, 6.39738, 6.39738, 6.39738, 6.39738, 6.39738, 6.39738, 6.39738, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.82392, 6.10178, 6.10178, 6.10178, 6.10178, 6.10178, 6.10178, 6.10178, 6.10178, 6.10178, 6.57668, 6.57668, 6.57668, 6.57668, 6.57668, 6.57668, 6.57668, 6.57668, 6.57668, 6.57668, 6.90053, 6.90053, 6.90053, 6.90053, 6.90053, 6.90053, 6.90053, 6.90053, 6.90053, 6.90053, 6.90053, 6.90053, 5.80795, 5.80795, 5.80795, 5.80795, 5.80795, 5.80795, 5.80795, 5.80795, 6.00201, 6.00201, 6.00201, 6.00201, 6.00201, 6.00201, 6.00201, 6.00201, 6.00201, 6.00201, 6.00201, 6.35451, 6.35451, 6.35451, 6.35451, 6.35451, 6.35451, 6.35451, 6.35451, 6.35451, 6.35451, 6.35451, 6.35451, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.61786, 6.91492, 6.91492, 6.91492, 6.91492, 6.91492, 6.91492, 6.91492, 6.91492, 6.91492, 6.91492, 6.91492, 6.21836, 6.21836, 6.21836, 6.21836, 6.21836, 6.21836, 6.21836, 6.21836, 6.21836, 6.21836, 6.21836, 6.59154, 6.59154, 6.59154, 6.59154, 6.59154, 6.59154, 6.59154, 6.59154, 6.59154, 5.4932, 5.4932, 5.4932, 5.4932, 5.4932, 5.4932, 5.4932, 5.4932, 5.4932, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 5.99951, 6.45769, 6.45769, 6.45769, 6.45769, 6.45769, 6.45769, 6.45769, 6.45769, 6.45769, 6.45769, 6.45769, 6.45769, 6.79352, 6.79352, 6.79352, 6.79352, 6.79352, 6.79352, 6.79352, 6.79352, 6.79352, 5.8499, 5.8499, 5.8499, 5.8499, 5.8499, 5.8499, 5.8499, 5.8499, 5.8499, 5.8499, 6.23469, 6.23469, 6.23469, 6.23469, 6.23469, 6.23469, 6.23469, 6.23469, 6.23469, 6.23469, 6.23469, 6.50065, 6.50065, 6.50065, 6.50065, 6.50065, 6.50065, 6.50065, 6.50065, 6.50065, 6.50065, 6.67076, 6.67076, 6.67076, 6.67076, 6.67076, 6.67076, 6.67076, 6.67076, 6.67076, 6.67076, 6.67076, 6.79647, 6.79647, 6.79647, 6.79647, 6.79647, 6.79647, 6.79647, 6.79647, 6.79647, 6.79647, 6.79647, 6.79647, 7.31586, 7.31586, 7.31586, 7.31586, 7.31586, 7.31586, 7.31586, 7.31586, 7.31586, 7.31586, 7.31586, 7.3025, 7.3025, 7.3025, 7.3025, 7.3025, 7.3025, 7.3025, 7.3025, 7.3025, 7.3025]
LATITUDES = [46.17685, 46.17685, 46.17685, 46.17685, 46.17685, 46.17685, 46.17685, 46.17685, 46.17685, 46.17685, 46.17685, 45.89494, 45.89494, 45.89494, 45.89494, 45.89494, 45.89494, 45.89494, 45.89494, 45.89494, 45.89494, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.89794, 45.65578, 45.65578, 45.65578, 45.65578, 45.65578, 45.65578, 45.65578, 45.65578, 45.65578, 45.65756, 45.65756, 45.65756, 45.65756, 45.65756, 45.65756, 45.65756, 45.65756, 45.65756, 45.65756, 45.54313, 45.54313, 45.54313, 45.54313, 45.54313, 45.54313, 45.54313, 45.54313, 45.54313, 45.54313, 45.54313, 45.54313, 45.37753, 45.37753, 45.37753, 45.37753, 45.37753, 45.37753, 45.37753, 45.37753, 45.27395, 45.27395, 45.27395, 45.27395, 45.27395, 45.27395, 45.27395, 45.27395, 45.27395, 45.27395, 45.27395, 45.32783, 45.32783, 45.32783, 45.32783, 45.32783, 45.32783, 45.32783, 45.32783, 45.32783, 45.32783, 45.32783, 45.32783, 45.411, 45.411, 45.411, 45.411, 45.411, 45.411, 45.411, 45.411, 45.411, 45.411, 45.411, 45.411, 45.411, 45.26072, 45.26072, 45.26072, 45.26072, 45.26072, 45.26072, 45.26072, 45.26072, 45.26072, 45.26072, 45.26072, 45.11517, 45.11517, 45.11517, 45.11517, 45.11517, 45.11517, 45.11517, 45.11517, 45.11517, 45.11517, 45.11517, 45.01923, 45.01923, 45.01923, 45.01923, 45.01923, 45.01923, 45.01923, 45.01923, 45.01923, 45.00409, 45.00409, 45.00409, 45.00409, 45.00409, 45.00409, 45.00409, 45.00409, 45.00409, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.94609, 44.83699, 44.83699, 44.83699, 44.83699, 44.83699, 44.83699, 44.83699, 44.83699, 44.83699, 44.83699, 44.83699, 44.83699, 44.77139, 44.77139, 44.77139, 44.77139, 44.77139, 44.77139, 44.77139, 44.77139, 44.77139, 44.69552, 44.69552, 44.69552, 44.69552, 44.69552, 44.69552, 44.69552, 44.69552, 44.69552, 44.69552, 44.70565, 44.70565, 44.70565, 44.70565, 44.70565, 44.70565, 44.70565, 44.70565, 44.70565, 44.70565, 44.70565, 44.57217, 44.57217, 44.57217, 44.57217, 44.57217, 44.57217, 44.57217, 44.57217, 44.57217, 44.57217, 44.44757, 44.44757, 44.44757, 44.44757, 44.44757, 44.44757, 44.44757, 44.44757, 44.44757, 44.44757, 44.44757, 44.12458, 44.12458, 44.12458, 44.12458, 44.12458, 44.12458, 44.12458, 44.12458, 44.12458, 44.12458, 44.12458, 44.12458, 44.12649, 44.12649, 44.12649, 44.12649, 44.12649, 44.12649, 44.12649, 44.12649, 44.12649, 44.12649, 44.12649, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39]





