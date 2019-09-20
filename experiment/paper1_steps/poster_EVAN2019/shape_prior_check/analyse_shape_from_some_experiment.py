import matplotlib.pyplot as plt

# Load shapes
lines = []
with open('shape_from_some_experiment.txt') as f:
    for l in f:
        lines.append(float(l.split('\n')[0]))
# Build his
plt.hist(lines, bins=50,  histtype='step')
plt.show()