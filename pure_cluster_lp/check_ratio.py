import math
import csv
from collections import defaultdict
import numpy as np
import time
import itertools

base = 200
target = 1.484
pset = target / 2
ppivot = 1 - pset

##############################
###Algorithm parameter
##############################
a = 0.40
b = 0.57

ca = 0
cb = 0.8

alpha = target / (2 - target)

# a = 0.19	
# b = 0.57
# c = 0.37
# pc = math.pow((c - b)/(a - b), 2)
# d = (1 - c) / (1 - pc)
EDGES = [[True, True, True], [True, True, False], [True, False, True], \
[False, True,  True], [True, False, False], [False, True, False],[False, False , True], [False, False, False]]

def prob(x, positive):
	if positive:
		global a
		global b 
		if x < a:
			return 0
		elif x <= b:
			return x
		else:
			return x
	else:
		return x * x

def coefficent(x, p, positive):
	if positive:
		return (target - pset * 2 / (1 + x)) / ppivot
	else:
		return (target - pset / (1 + x)) / ppivot


def edgealg(x, y, z, p, edges):
	pvw = prob(y, edges[1])
	puw = prob(z, edges[2])
	if edges[1] and edges[2] and y >= a and z >= a and y <= b and z <= b:
		if edges[0]:
			return ((1 - y) + (1 - z) - 2 * p)
		else:
			return p
	else:
		if edges[0]:
			return puw + pvw - 2 * puw * pvw
		else:
			return (1 - puw) * ( 1 - pvw)

def covariance(x, y, z, p, edges):
	return p - (1 - y)*(1 - z)

def edgelp(x, y, z, p, edges):
	pvw = prob(y, edges[1])
	puw = prob(z, edges[2])
	if edges[1] and edges[2] and y > a and z > a and y < b and z < b:
		if edges[0]:
			return x * ((1 - y) + (1 - z) - p)
		else:
			return (1 - x) * ((1 - y) + (1 - z) - p)
	else:
		if edges[0]:
			return x * (1 - puw * pvw)
		else:
			return (1 - x) *(1 - puw * pvw)

def ratio(x, y, z, p, edges):
	s1 = edgealg(x, y, z, p, edges) + edgealg(z, y, x, p, [edges[2], edges[1], edges[0]]) + edgealg(y, x, z, p, [edges[1], edges[0], edges[2]])
	s2 = coefficent(x, p, edges[0]) * edgelp(x, y, z, p, edges) + coefficent(z, p, edges[2])* edgelp(z, y, x, p, [edges[2], edges[1], edges[0]]) + \
	 coefficent(y, p, edges[1]) * edgelp(y, x, z, p, [edges[1], edges[0], edges[2]])
	# print(s1, s2, coefficent(x, p, edges[0]), edgelp(x, y, z, p, edges), coefficent(z, p, edges[2]), edgelp(z, y, x, p, [edges[2], edges[1], edges[0]]), \
	# 	coefficent(y, p, edges[1]),edgelp(y, x, z, p, [edges[1], edges[0], edges[2]]))
	return s1 - s2

def check1(a, b, c):
	return a + b < c

def createtriangles(x, y, z, p):
	f = [0] * len(EDGES)
	for i in range(len(EDGES)):
		f[i] = ratio(x, y, z, p, EDGES[i])
	idx = np.argmax(f)
	edges = EDGES[idx]
	covxy, covxz, covyz = covariance(z, y, x, p, [edges[2], edges[1], edges[0]]), \
		covariance(y, x, z, p, [edges[1], edges[0], edges[2]]), covariance(x, y, z, p, edges)
	if f[idx] > 0:
		print(">= 0 |", x, y, z, p, "\n type:", idx, "ratio:", f[idx], covxy, covxz, covyz)

if __name__ == '__main__':
	start_time = time.time()
	f = [0, 0, 0, 0]

	print("ratio: ", target)

	# createtriangles(0.5, 0.5, 1, 0)
	# createtriangles(0.392, 0.392, 0.392, 0.216)
	# createtriangles(0.42, 0.42, 0.42, 0.16)
	# xRange, yRange, zRange, pRange = [0.38, 0.39], [0.6, 0.61], [0.98, 1], [0, 0]
	# Range = [0.31, 0.32, 0.31, 0.32, 0.63, 0.64, 0.36, 0.3605]
	# Range = [0.33001, 0.34, 0.33001, 0.34, 0.33001, 0.34, 0.51, 0.515]
	# Range = [0.32001, 0.32999999999, 0.32001, 0.32999999999, 0.32001, 0.32999999999, 0.505, 0.515]
	# Range = [0.31001, 0.32, 0.31001, 0.32, 0.31001, 0.32, 0.52, 0.535]
	# Range = [a - 0.005, a - 0.000000001, a - 0.005, a - 0.000000001, 2 * a - 0.01, 2 * a, 1 - 2*a, 1 - 2*a + 0.01]
	# Range = [0.39, 0.399999999999, 0.5599999999999999, 0.569999999999, 0.96, 0.97, 0.030000000000000027, 0.031000000000000028]
	Range = [a - 0.01, a - 0.000000001, a - 0.01, a - 0.000000001, 2 * a - 0.01, 2 * a, 1 - 2*a, 1 - 2*a + 0.01]
	# Range = [a + 0.00000001, a + 0.005, a + 0.00000001, a + 0.005, a + 0.00000001, a + 0.005, 2.0 / 3 - a - 0.005, 2.0 / 3 - a]
	# Range = [0.32, 0.33, 0.65, 0.66, 0.98, 0.99, 0.0050000000000000044, 0.020000000000000018]
	xRange, yRange, zRange, pRange = Range[0:2], Range[2:4], Range[4:6], Range[6:8]
	for x, y, z, p in itertools.product(xRange, yRange, zRange, pRange):
		createtriangles(x, y, z, p)

	# createtriangles(0.45, 0.45, 0.9, 0)
	# for a1 in range(0, base+1):
	# 	for a2 in range(a1, base+1):
	# 		for a3 in range(a2, min(base + 1, a1 + a2 + 1)):
	# 			x, y, z = 1.0 * a1 / base, 1.0 * a2 / base, 1.0 * a3 / base
	# 			if check1(x, y, z) or check1(z, x, y) or check1(y, z, x):
	# 				continue
	# 			plower, pupper = max(0, 1 - x - y, 1 - y - z, 1- x-z), min(1 - x, 1 - y, 1-z)
	# 			# plower, pupper = max(0, 2 *(x + y + z) / 3 - 1, 1 - x - y, 1 - y - z, 1- x-z), min(1 - x, 1 - y, 1-z)
	# 			if plower > pupper:
	# 				continue
	# 			createtriangles(x, y, z, plower)
	# 			if plower == pupper:
	# 				continue
	# 			createtriangles(x, y, z, pupper)

	print("--- %s seconds ---" % (time.time() - start_time))
