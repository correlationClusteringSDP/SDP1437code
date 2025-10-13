import math
import csv
from collections import defaultdict
import numpy as np
import time
import itertools

base = 200
target = 1.45
pset = target / 2
ppivot = 1 - pset

##############################
###Algorithm parameter
##############################
a = 0.32
b = 1.1

na = 0.6
nb = 0.84

independent = 0 # 0.22

alpha = target / (2 - target)

# a = 0.19	
# b = 0.57
# c = 0.37
# pc = math.pow((c - b)/(a - b), 2)
# d = (1 - c) / (1 - pc)
EDGES = [[True, True, True], [True, True, False], [True, False, True], \
[False, True,  True], [True, False, False], [False, True, False],[False, False , True], [False, False, False]]

def prob(x, positive):
	# if x <= a:
	# 	return 0
	# elif x <= b:
	# 	return 1
	if positive:
		global a
		global b 
		if x < a:
			return 0
		elif x <= b:
			return x
		else:
			return 1
	else:
		# if x < na:
		# 	return x
		# elif x <= nb:
		# 	return (x - na ) * (x - na) / (nb - na) + na
		# else:
		if x >= 0.97:
			return 1
		else:
			return x * x


## independent rounding
# def indep_prob(x, positive):
# 	if positive:
# 		global a
# 		global b 
# 		if x < a:
# 			return 0
# 		elif x <= b:
# 			return (x - a) * (x -a) / ((b - a) * (b - a))
# 		else:
# 			return 1
# 	else:
# 		if x < na:
# 			return x
# 		elif x <= nb:
# 			return (x - na ) * (x - na) / (nb - na) + na
# 		else:
# 			return x

def coefficent(x, p, positive):
	# return 3
	if positive:
		return (target - pset * 2 / (1 + x)) / ppivot
	else:
		return (target - pset / (1 + x)) / ppivot


def edgealg(x, y, z, p, edges):
	pvw = prob(y, edges[1])
	puw = prob(z, edges[2])

	dePosCost, inPosCost =  (1 - y) + (1 - z) - 2 * p, puw + pvw - 2 * puw * pvw
	deNegCost, inNegCost = p, (1 - puw) * ( 1 - pvw)
	if edges[1] and edges[2] and y > a and z > a and y <= b and z <= b:
		if edges[0]:
			return independent * inPosCost + (1 - independent) * dePosCost
		else:
			return independent * inNegCost + (1 - independent) * deNegCost
	else:
		if edges[0]:
			return inPosCost
		else:
			return inNegCost

def covariance(x, y, z, p, edges):
	return p - (1 - y)*(1 - z)

def edgelp(x, y, z, p, edges):
	pvw = prob(y, edges[1])
	puw = prob(z, edges[2])

	dePosLp, inPosLp = x * ((1 - y) + (1 - z) - p), x * (1 - puw * pvw)
	deNegLp, inNegLp = (1 - x) * ((1 - y) + (1 - z) - p), (1 - x) *(1 - puw * pvw)
	if edges[1] and edges[2] and y > a and z > a and y <= b and z <= b:
		if edges[0]:
			return independent * inPosLp + (1 - independent) * dePosLp
		else:
			return independent * inNegLp + (1 - independent) * deNegLp
	else:
		if edges[0]:
			return inPosLp
		else:
			return inNegLp

def ratio(x, y, z, p, edges):
	s1 = edgealg(x, y, z, p, edges) + edgealg(z, y, x, p, [edges[2], edges[1], edges[0]]) + edgealg(y, x, z, p, [edges[1], edges[0], edges[2]])
	s2 = coefficent(x, p, edges[0]) * edgelp(x, y, z, p, edges) + coefficent(z, p, edges[2])* edgelp(z, y, x, p, [edges[2], edges[1], edges[0]]) + \
	 coefficent(y, p, edges[1]) * edgelp(y, x, z, p, [edges[1], edges[0], edges[2]])
	# if s1 - s2 > 0:
	# 	print(s1, s2, coefficent(x, p, edges[0]), edgelp(x, y, z, p, edges), coefficent(z, p, edges[2]), edgelp(z, y, x, p, [edges[2], edges[1], edges[0]]), \
	# 		coefficent(y, p, edges[1]),edgelp(y, x, z, p, [edges[1], edges[0], edges[2]]))
	return s1 - s2

def check1(a, b, c):
	return a + b < c

def createtriangles(x, y, z, p):
	ret = True

	z = min(x + y, z)
	f = [0] * len(EDGES)
	for i in range(len(EDGES)):
		f[i] = ratio(x, y, z, p, EDGES[i])
		# print(x, y, z, p, i, f[i])
	idx = np.argmax(f)
	edges = EDGES[idx]
	covxy, covxz, covyz = covariance(z, y, x, p, [edges[2], edges[1], edges[0]]), \
		covariance(y, x, z, p, [edges[1], edges[0], edges[2]]), covariance(x, y, z, p, edges)
	if f[idx] > 0:
		ret = False
		print(">= 0 |", x, y, z, p, "\n type:", idx, "ratio:", f[idx], covxy, covxz, covyz)

	return ret

def check_range(Range, base2 = 10):
	xRange, yRange, zRange, pRange = Range[0:2], Range[2:4], Range[4:6], Range[6:8]

	xd, yd, zd, pd = xRange[1] - xRange[0], yRange[1] - yRange[0], zRange[1] - zRange[0], pRange[1] - pRange[0]
	for a1 in range(0, base2+1):
		for a2 in range(0, base2 + 1):
			for a3 in range(0, base2 + 1):
				for a4 in range(0, base2 + 1):
					dx, dy, dz, dp = xd * a1 / base2, yd * a2 / base2, zd * a3 / base2, pd * a4 / base2
					x, y, z, p = xRange[0] + dx, yRange[0] + dy, zRange[0] + dz, pRange[0] + dp
					if check1(x, y, z) or check1(z, x, y) or check1(y, z, x):
						continue
					createtriangles(x, y, z, p)
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
	# Range = [0.00, 0.01, 0.99, 1.0, 0.99, 1.0, 0.0, 0.010000000000000009]
	# Range = [0.01, 0.05, 0.95, 0.98, 0.95, 1.0, 0.0, 0.01]
	Range = [0.31, 0.32, 0.65, 0.7, 0.96, 0.98, 0.0, 0.040000000000000036]
	# Range = [0.005, 0.01, 0.005, 0.01, 0.005, 0.01, 0.925, 0.995]
	# Range = [0.0, 0.02, 0.0, 0.02, 0.0, 0.02, 0.925, 1.0]

	check_range(Range)
	# Range = [a - 0.005, a - 0.000000001, a - 0.005, a - 0.000000001, 2 * a - 0.01, 2 * a, 1 - 2*a, 1 - 2*a + 0.01]
	# Range = [a - 0.01, a - 0.000000001, a - 0.01, a - 0.000000001, 2 * a - 0.01, 2 * a, 1 - 2*a, 1 - 2*a + 0.01]
	# Range = [a + 0.00000001, a + 0.005, a + 0.00000001, a + 0.005, a + 0.00000001, a + 0.005, 1 - 3* a / 2 - 0.0075, 1 - 3*a / 2]
	# Range = [0.32, 0.33, 0.65, 0.66, 0.98, 0.99, 0.0050000000000000044, 0.020000000000000018]



	# createtriangles(0.39, 0.61, 1.0, 0)

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
