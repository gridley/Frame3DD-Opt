#!/usr/bin/env python3
# Given a few trios of certain files, this plots the shapes of trusses, potentially overlapping with each other.
import matplotlib.pyplot as plt
import numpy as np
import sys
num_args = len(sys.argv)-1
num_designs = int(num_args / 3)
if num_args % 3 > 0:
    raise Exception('Incorrect number of commands provided')

def parse_nodes(filename):
    nodes = []
    with open(filename) as fh:
        lines = fh.readlines()
        i = 1
        for line in lines[1:]:
            i += 1
            if line.startswith('#'):
                continue
            else:
                split_line = line.split()
                if split_line:
                    num_nodes = int(split_line[0])
                    break
        for line in lines[i:]:
            split_line = line.split()
            if split_line and not line.startswith('#'):
                nodes.append((float(split_line[1]), float(split_line[2]), float(split_line[3])))
            if len(nodes) == num_nodes:
                return nodes

lines = []
for i in range(num_designs):
    connectivity = np.loadtxt(sys.argv[3*i+1], dtype=np.int32) - 1
    nodes = parse_nodes(sys.argv[3*i+2])
    areas = np.loadtxt(sys.argv[3*i+3])

    connection = connectivity[0]
    line = plt.plot((nodes[connectivity[0][0]][0], nodes[connectivity[0][1]][0]),
                    (nodes[connectivity[0][0]][1], nodes[connectivity[0][1]][1]), linewidth=areas[0]**.5)
    lines.append(line) # for legend later
    for i, connection in enumerate(connectivity[1:]):
        if areas[i+1] > 1e-3:
            plt.plot((nodes[connection[0]][0], nodes[connection[1]][0]),
                            (nodes[connection[0]][1], nodes[connection[1]][1]), linewidth=areas[i+1]**.5, c=line[0].get_color())

plt.savefig('truss_shape.png')
