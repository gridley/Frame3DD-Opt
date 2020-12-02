#!/usr/bin/env python
# Plots several differential evolution convergence
# trends. You should pass pairs of names to go in the
# legend, then names of files containing convergence history.
import matplotlib.pyplot as plt
import numpy as np
import sys

lines = []
names = []
num_args = len(sys.argv) - 1
num_files = int(num_args / 2)

for i in range(num_files):
    name = sys.argv[2*i+1]
    names.append(name)
    filename = sys.argv[2*i+2]
    data = np.loadtxt(filename)
    line = plt.plot(data[:,0], data[:, 1])
    lines.append(line[0])
    plt.fill_between(data[:,0], data[:,3] - data[:,4], data[:,3] + data[:,4], color=line[0].get_color(), alpha=0.4)

plt.legend(lines, names)
plt.title('Differential Evolution Convergence')
plt.xlabel('Evolution Iteration')
plt.ylabel('Truss volume')
plt.savefig('convergence.png')
plt.clf()
plt.cla()

# Do the same, but only for the second half of the calculation
midpoint = int(data.shape[0] / 2)
lines = []
names = []
for i in range(num_files):
    name = sys.argv[2*i+1]
    names.append(name)
    filename = sys.argv[2*i+2]
    data = np.loadtxt(filename)
    line = plt.plot(data[midpoint:,0], data[midpoint:, 1])
    lines.append(line[0])
    plt.fill_between(data[midpoint:,0], data[midpoint:,3] - data[midpoint:,4], data[midpoint:,3] + data[midpoint:,4], color=line[0].get_color(), alpha=0.4)

plt.legend(lines, names)
plt.title('Differential Evolution Convergence')
plt.xlabel('Evolution Iteration')
plt.ylabel('Truss volume')
plt.savefig('convergence_second_half.png')
