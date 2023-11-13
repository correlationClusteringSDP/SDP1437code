import math
import csv
from collections import defaultdict
import numpy as np
import time
import bisect

from functools import reduce

##############################
###Algorithm parameter
##############################

posThreshold1 = 1/3
posThreshold2 = 1
# negThreshold1 = -1
# negThreshold2 = -1

EDGES = [[True, True, True], [True, True, False], [True, False, True], \
[False, True,  True], [True, False, False], [False, True, False],[False, False , True], [False, False, False]]

target = 1.437
pset = target / 2
ppivot = 1 - pset
alpha = target / (2 - target)

base = 10
base2 = 10
base3 = 10
base4 = 10
oneoverbase = 1 / base
# errorForSet = oneoverbase * oneoverbase * 3
record = set()
edgesIndex = {}
splittingPoints = []
epsilon = 0.00

splittingPoints = []
splittingPoints.append(posThreshold1)
splittingPoints.append(posThreshold2)
# splittingPoints = list(reduce(lambda x, y: x + y, Threshold))
splittingPoints += [i / base for i in range(0, base + 1)]

splittingPoints += [0.15, 0.62, 0.65]
# splittingPoints += [0.05, 0.95, 0.51, 0.49, 0.99]/
splittingPoints += [i / 100 for i in range(32, 60, 2)]
splittingPoints += [i / 100 for i in range(42, 54, 1)]
# splittingPoints += [0.45 + 0.1 * i / base3 for i in range(0, base3, 1)]
# splittingPoints += [i / 100 for i in range(38, 56, 1)]
splittingPoints += [i / 100 for i in range(95, 100, 1)]
# splittingPoints += [0.95 + 0.1 * i / base3 for i in range(0, base3, 1)]

class pivotAnalysis:
	def __init__(self):
		self.posThreshold1 = posThreshold1
		self.posThreshold2 = posThreshold2
		self.negCorrelatedRoundingThreshold1 = 1.1
		self.negCorrelatedRoundingThreshold2 = 1.1
		self.posCorrelatedRoundingThreshold1 = self.posThreshold1
		self.posCorrelatedRoundingThreshold2 = self.posThreshold2

	def set(self, val):
		self.posThreshold1 = val[0]
		self.posThreshold2 = val[1]
		self.negCorrelatedRoundingThreshold1 = val[2]
		self.negCorrelatedRoundingThreshold2 = val[3]
		self.posCorrelatedRoundingThreshold1 = val[4]
		self.posCorrelatedRoundingThreshold2 = val[5]

	def correlatedRounding(self, x, positive):
		if positive:
			if x >= self.posCorrelatedRoundingThreshold1 and x < self.posCorrelatedRoundingThreshold2:
				return True
			else:
				return False
		else:
			if x >= self.negCorrelatedRoundingThreshold1 and x < self.negCorrelatedRoundingThreshold2:
				return True
			else:
				return False

	def prob(self, x, positive):
		if positive:
			if x >= self.posThreshold1 and x < self.posThreshold2:
				return x
			elif x >= self.posThreshold2:
				return 1
			else:
				return 0
		else:
			# if x <= 0.2:
			# 	return 0
			# # elif x < 0.8:
			# # 	return x
			# else:
			return x
			# return x

	def probCorrelatedInput(self, x, xinput, positive):
		if positive:
			if xinput >= self.posThreshold1 and xinput < self.posThreshold2:
				return x
			elif xinput >= self.posThreshold2:
				return 1
			else:
				return 0
		else:
			return x
			
	def edgealgCorrelatedInput(self, x, y, z, p, yinput, zinput, edges):
		pvw = self.probCorrelatedInput(y, yinput, edges[1])
		puw = self.probCorrelatedInput(z, zinput, edges[2])

		# print("edgealg:", yinput, zinput, self.correlatedRounding(yinput, edges[1]), self.correlatedRounding(zinput, edges[2]))
		if self.correlatedRounding(yinput, edges[1]) and self.correlatedRounding(zinput, edges[2]):
			# print(x, y, z, p, edges, "correlatedRounding",  yinput, zinput)
			if edges[0]:
				return ((1 - y) + (1 - z) - 2 * p)
			else:
				return p
		else:
			if edges[0]:
				return puw + pvw - 2 * puw * pvw
			else:
				return (1 - puw) * ( 1 - pvw)

	def edgelpCorrelatedInput(self, x, y, z, p, yinput, zinput, edges):
		pvw = self.probCorrelatedInput(y, yinput, edges[1])
		puw = self.probCorrelatedInput(z, zinput, edges[2])

		if self.correlatedRounding(yinput, edges[1]) and self.correlatedRounding(zinput, edges[2]):
			if edges[0]:
				return x * ((1 - y) + (1 - z) - p)
			else:
				return (1 - x) * ((1 - y) + (1 - z) - p)
		else:
			if edges[0]:
				return x * (1 - puw * pvw)
			else:
				return (1 - x) *(1 - puw * pvw)

	def edgealg(self, x, y, z, p, edges):
		pvw = self.prob(y, edges[1])
		puw = self.prob(z, edges[2])

		if self.correlatedRounding(y, edges[1]) and self.correlatedRounding(z, edges[2]):
			# print(x, y, z, p, edges, "correlatedRounding", self.posCorrelatedRoundingThreshold1, self.posCorrelatedRoundingThreshold2)
			if edges[0]:
				return ((1 - y) + (1 - z) - 2 * p)
			else:
				return p
		else:
			if edges[0]:
				return puw + pvw - 2 * puw * pvw
			else:
				return (1 - puw) * ( 1 - pvw)

	def edgelp(self, x, y, z, p, edges):
		pvw = self.prob(y, edges[1])
		puw = self.prob(z, edges[2])
		if self.correlatedRounding(y, edges[1]) and self.correlatedRounding(z, edges[2]):
			if edges[0]:
				return x * ((1 - y) + (1 - z) - p)
			else:
				return (1 - x) * ((1 - y) + (1 - z) - p)
		else:
			if edges[0]:
				return x * (1 - puw * pvw)
			else:
				return (1 - x) *(1 - puw * pvw)

def filtersplittlingpoints(x):
	# interval = [[0, 0.16], [posThreshold1, posThreshold1], [0.38, 0.65], \
	# [posThreshold2, posThreshold2], [0.9, 1.0]]
	interval = [[0, 1]]
	for left, right in interval:
		if x >= left and x <= right:
			return True
	return False

def roundIndex(x, positive):
	return round(int((x) * 10000))
	# return round(int(x * 10000))

def psetEdgeAlgCost(x, positive):
	if positive:
		return 2 * x / (1 + x) + min(oneoverbase, x) * oneoverbase
	else:
		return (1 - x) / (1 + x)

def fpivotfix(x, delta, positive):
	if positive:
		if x == 0:
			return 0
		else:
			# return 2 * alpha  * x / (1 + x)
			return 2 * alpha  * (x / (1 + x) - delta*delta / (2 * x) )
	else:
		return alpha * (1 + 2*x) / (1 + x)

def psetRatio(x, y, z, edges):
	s1 = psetEdgeAlgCost(x, edges[0]) + psetEdgeAlgCost(y, edges[1]) + psetEdgeAlgCost(z, edges[2])
	s2 = psetEdgelp(x, edges[0]) + psetEdgelp(y, edges[1]) + psetEdgelp(z, edges[2])
	# return s1 - s2 + errorForSet
	return s1 - target * s2

# def ppivotEdgeAlg15Ratio(x, y, z, edges):
# 	s1 = feasibleDP.fpivot(x, edges[0])*psetEdgelp(x, edges[0]) + feasibleDP.fpivot(y, edges[1])*psetEdgelp(y, edges[1]) + feasibleDP.fpivot(z, edges[2])* psetEdgelp(z, edges[2])
# 	s2 = target * psetEdgelp(x, edges[0]) + target * psetEdgelp(y, edges[1]) + target * psetEdgelp(z, edges[2])
# 	return s1 - s2

# def ppivotEdgeAlgTargetRatio(x, y, z, p, edges):
# 	s1 = edgealg(x, y, z, p, edges) + edgealg(z, y, x, p, [edges[2], edges[1], edges[0]]) + edgealg(y, x, z, p, [edges[1], edges[0], edges[2]])
# 	s2 = target * psetEdgelp(x, edges[0]) + target * psetEdgelp(y, edges[1]) + target * psetEdgelp(z, edges[2])
# 	return s1 - s2

def ratio1(x, y, z, p, xRange, yRange, zRange, edges):
	xinput, yinput, zinput = xRange[0], yRange[0], zRange[0]
	xLen, yLen, zLen = xRange[1] - xRange[0],  yRange[1] - yRange[0], zRange[1] - zRange[0]
	s1 = pa.edgealgCorrelatedInput(x, y, z, p, yinput, zinput, edges) + \
	pa.edgealgCorrelatedInput(z, y, x, p, yinput, xinput, [edges[2], edges[1], edges[0]]) + \
	pa.edgealgCorrelatedInput(y, x, z, p, xinput, zinput, [edges[1], edges[0], edges[2]])

	# s2 = dp.fpivot(x, edges[0]) * pa.edgelpCorrelatedInput(x, y, z, p, yinput, zinput, edges) + \
	# dp.fpivot(z, edges[2])*  pa.edgelpCorrelatedInput(z, y, x, p, yinput, xinput, [edges[2], edges[1], edges[0]]) + \
	# dp.fpivot(y, edges[1]) * pa.edgelpCorrelatedInput(y, x, z, p, xinput, zinput, [edges[1], edges[0], edges[2]])

	s2 = fpivotfix(x, xLen, edges[0]) * pa.edgelpCorrelatedInput(x, y, z, p, yinput, zinput, edges) + \
	fpivotfix(z, yLen, edges[2])*  pa.edgelpCorrelatedInput(z, y, x, p, yinput, xinput, [edges[2], edges[1], edges[0]]) + \
	fpivotfix(y, zLen, edges[1]) * pa.edgelpCorrelatedInput(y, x, z, p, xinput, zinput, [edges[1], edges[0], edges[2]])
	# s1 = pa.edgealgCorrelatedInput(x, y, z, p, yinput, zinput, edges) + \
	# pa.edgealgCorrelatedInput(z, y, x, p, yinput, xinput, [edges[2], edges[1], edges[0]]) + \
	# pa.edgealgCorrelatedInput(y, x, z, p, xinput, zinput, [edges[1], edges[0], edges[2]])
	# s2 = pa.edgelpCorrelatedInput(x, y, z, p, yinput, zinput, edges) + \
	# pa.edgelpCorrelatedInput(z, y, x, p, yinput, xinput, [edges[2], edges[1], edges[0]]) + \
	# pa.edgelpCorrelatedInput(y, x, z, p, xinput, zinput, [edges[1], edges[0], edges[2]])
	# s2 = target * s2
	# s1 = edgealg(x, y, z, p, edges) + edgealg(z, y, x, p, [edges[2], edges[1], edges[0]]) + edgealg(y, x, z, p, [edges[1], edges[0], edges[2]])
	# s2 = fpivotfix(x, edges[0]) * edgelp(x, y, z, p, edges) + fpivotfix(z, edges[2])* \
	# edgelp(z, y, x, p, [edges[2], edges[1], edges[0]]) + fpivotfix(y, edges[1]) * edgelp(y, x, z, p, [edges[1], edges[0], edges[2]])
	# error = int(edges[0]) * xLen * xLen + int(edges[1]) * yLen * yLen + int(edges[2]) * zLen * zLen 
	# error /= 2
	# print(s1, s2)
	return s1 - s2 

def ratio(x, y, z, p, xRange, yRange, zRange, edges):
	xinput, yinput, zinput = xRange[0], yRange[0], zRange[0]
	xLen, yLen, zLen = 0, 0, 0
	s1 = pa.edgealgCorrelatedInput(x, y, z, p, yinput, zinput, edges) + \
	pa.edgealgCorrelatedInput(z, y, x, p, yinput, xinput, [edges[2], edges[1], edges[0]]) + \
	pa.edgealgCorrelatedInput(y, x, z, p, xinput, zinput, [edges[1], edges[0], edges[2]])

	s2 = fpivotfix(x, xLen, edges[0]) * pa.edgelpCorrelatedInput(x, y, z, p, yinput, zinput, edges) + \
	fpivotfix(z, yLen, edges[2])*  pa.edgelpCorrelatedInput(z, y, x, p, yinput, xinput, [edges[2], edges[1], edges[0]]) + \
	fpivotfix(y, zLen, edges[1]) * pa.edgelpCorrelatedInput(y, x, z, p, xinput, zinput, [edges[1], edges[0], edges[2]])

	return s1 - s2 

# def ratio2(x, y, z, p, edges, pa):
# 	s1 = pa.edgealg(x, y, z, p, edges) + pa.edgealg(z, y, x, p, [edges[2], edges[1], edges[0]]) + pa.edgealg(y, x, z, p, [edges[1], edges[0], edges[2]])
# 	# s2 = target * pa.edgelp(x, y, z, p, edges) + target * pa.edgelp(z, y, x, p, [edges[2], edges[1], edges[0]]) + \
# 	# target * pa.edgelp(y, x, z, p, [edges[1], edges[0], edges[2]])
# 	s2 = fpivotfix(x, edges[0]) * pa.edgelp(x, y, z, p, edges) + fpivotfix(z, edges[2])* pa.edgelp(z, y, x, p, [edges[2], edges[1], edges[0]]) + \
# 	 fpivotfix(y, edges[1]) * pa.edgelp(y, x, z, p, [edges[1], edges[0], edges[2]])

# 	# print(s1, s2, coefficent(x, p, edges[0]), edgelp(x, y, z, p, edges), coefficent(z, p, edges[2]), edgelp(z, y, x, p, [edges[2], edges[1], edges[0]]), \
# 	# 	coefficent(y, p, edges[1]),edgelp(y, x, z, p, [edges[1], edges[0], edges[2]]))
# 	return s1 - s2

def check1(a, b, c):
	return a + b < c

def check2(a, b, c, p):
	return a + b + c + 2 * p < 2 or a + b + p < 1 or a + c + p < 1 or b + c + p < 1 or a + p > 1 or b + p > 1 or c + p >1

def check3(a, b, c):
	return 0.5 - ita <= a <= 0.5 + ita and 0.5 - ita <= b <= 0.5 + ita and ( c <= 0.5 + 5*ita or c >= 1 - ita)

# def checkzero(x, y, z, edges):
# 	if edges[1] is False and edges[2] is False and y >= 0.99999 and z >= 0.99999:
# 		return True

# def roundIndex(x, positive):
# 	return round((x + 2 * int(positive)) * 10000)

def checkDuplicate(x, y, z, p, idx, idy, idz, edges):
	val = [(idx, roundIndex(x, edges[0])), (idy, roundIndex(y, edges[1])), (idz, roundIndex(z, edges[2]))] 
	val.sort()
	val.append((roundIndex(p, True)))
	val = tuple(val)
	if val in record:
		return True
	else:
		record.add(val)
		return False

def addEdgeIndex(x, positive):
	idx = roundIndex(x, positive)
	if idx not in edgesIndex:
		edgesIndex[idx] = len(edgesIndex)

def covariance(x, y, z, p):
	return p - (1 - y)*(1 - z)

def covarianceXX(x, xRange):
	return 1 - x

# def createtriangles(xRange, yRange, zRange, index):
# 	for i in range(len(EDGES)):
# 		edges = EDGES[i]
# 		addEdgeIndex(xRange[0], edges[0])
# 		addEdgeIndex(yRange[0], edges[1])
# 		addEdgeIndex(zRange[0], edges[2])
# 		idx, idy, idz = edgesIndex[roundIndex(xRange[0], edges[0])], \
# 		edgesIndex[roundIndex(yRange[0], edges[1])], edgesIndex[roundIndex(zRange[0], edges[2])]
# 		plower = max(0, 1- (xRange[1] + yRange[1] +zRange[1]) / 2, 1 - (xRange[1] + yRange[1]))
# 		pupper = min(1, 1 - xRange[0], 1 - yRange[0], 1 - zRange[0])
# 		for x in xRange:
# 			for y in yRange:
# 				for z in zRange:
# 					for p in [plower, pupper]:
# 						if checkDuplicate(x, y, z, p, idx, idy, idz, edges):
# 							continue
# 						# record.add((roundIndex(x, edges[0]), roundIndex(y, edges[1]), roundIndex(z, edges[2]), roundIndex(p, True)))
# 						# if checkzero(x, y, z, edges) or checkzero(y, x, z, [edges[1], edges[0], edges[2]]) or \
# 						# 	checkzero(z, y, x, [edges[2], edges[1], edges[0]]):
# 						# 	continue
# 						covxy, covxz, covyz = covariance(z, y, x, p, [edges[2], edges[1], edges[0]]), \
# 								covariance(y, x, z, p, [edges[1], edges[0], edges[2]]), covariance(x, y, z, p, edges)
# 						covxx, covyy, covzz = covarianceXX(x), covarianceXX(y), covarianceXX(z)
# 						# ps = psetRatio(x, y, z, edges)
# 						pp = ratio(x, y, z, p, edges)
# 						# pp = [0] * len(Threshold)
# 						# for j in range(len(Threshold)):
# 						# 	pa.set(Threshold[j])
# 						# 	pp[j] = ratio(x, y, z, p, xRange[0], yRange[0], zRange[0], edges, pa)
# 						# pa.set((0.31, 1.0, 0.25, 0.65))
# 						# pp15 = ratio2(x, y, z, p, edges, pa)
# 						# index.append((1 - x, 1 - y, 1 - z, p, i, covxy, covxz, covyz, ps, pp15) + tuple(pp))
# 						index.append((idx, idy, idz, i, x, y, z, p, xRange[0], yRange[0], zRange[0], covxy, covxz, covyz, covxx, covyy, covzz, pp))

def createtriangles(xRange, yRange, zRange, index):
	addEdgeIndex(xRange[0], True)
	addEdgeIndex(yRange[0], True)
	addEdgeIndex(zRange[0], True)
	idx, idy, idz = edgesIndex[roundIndex(xRange[0], True)], \
	edgesIndex[roundIndex(yRange[0], True)], edgesIndex[roundIndex(zRange[0], True)]

	f = [-100] * len(EDGES)
	xLen, yLen, zLen = xRange[1] - xRange[0], yRange[1] - yRange[0], zRange[1] - zRange[0]
	for a1 in range(0, base2 + 1):
		for a2 in range(0, base2 + 1):
			for a3 in range(0, base2 + 1):
				deltaX, deltaY, deltaZ = xLen * a1 / base2, yLen * a2 / base2, zLen * a3 / base2
				x, y, z = deltaX + xRange[0], deltaY + yRange[0], deltaZ + zRange[0]
				if check1(x, y, z) or check1(z, x, y) or check1(y, z, x):
					continue
				for p in [max(0, 1- (x + y +z)/2, 1- x - y), min(1 - x, 1 - y, 1-z)]:
					for i in range(len(EDGES)):
						temp = ratio(x, y, z, p, xRange, yRange, zRange, EDGES[i])
						# if f[i] < temp:
						# 	print("entry: ", x, y, z, p, xRange[0], yRange[0], zRange[0], EDGES[i], temp)
						f[i] = max(temp, f[i])
						

	costIdx = np.argmax(f)
	edges = EDGES[costIdx]
	plower = max(0, max(1- (xRange[1] + yRange[1] +zRange[1]) / 2, 1 - (xRange[1] + yRange[1]), 1 - (xRange[1] + zRange[1]), 1 - (zRange[1] + yRange[1]) ))
	pupper = min(1, min(1 - xRange[0], 1 - yRange[0], 1 - zRange[0]))
	for x in xRange:
		for y in yRange:
			for z in zRange:
				if x > y or x > z or y > z:
					continue
				for p in [plower, pupper]:
					if checkDuplicate(x, y, z, p, idx, idy, idz, [True, True, True]):
						continue
					# record.add((roundIndex(x, edges[0]), roundIndex(y, edges[1]), roundIndex(z, edges[2]), roundIndex(p, True)))
					# if checkzero(x, y, z, edges) or checkzero(y, x, z, [edges[1], edges[0], edges[2]]) or \
					# 	checkzero(z, y, x, [edges[2], edges[1], edges[0]]):
					# 	continue
					covxy, covxz, covyz = covariance(z, y, x, p), \
							covariance(y, x, z, p), covariance(x, y, z, p)
					covxx, covyy, covzz = covarianceXX(x, xRange), covarianceXX(y, yRange), covarianceXX(z, zRange)
					index.append((idx, idy, idz, costIdx, x, y, z, p, xRange[0], yRange[0], zRange[0], covxy, covxz, covyz, covxx, covyy, covzz, f[costIdx]))

def createtriangles2(xRange, yRange, zRange, index):
	addEdgeIndex(xRange[0], True)
	addEdgeIndex(yRange[0], True)
	addEdgeIndex(zRange[0], True)
	idx, idy, idz = edgesIndex[roundIndex(xRange[0], True)], \
	edgesIndex[roundIndex(yRange[0], True)], edgesIndex[roundIndex(zRange[0], True)]

	plower = max(0, max(1- (xRange[1] + yRange[1] +zRange[1]) / 2, 1 - (xRange[1] + yRange[1]), 1 - (xRange[1] + zRange[1]), 1 - (zRange[1] + yRange[1]) ))
	pupper = min(1, min(1 - xRange[0], 1 - yRange[0], 1 - zRange[0]))
	pLen = pupper - plower

	f = [[-100] * len(EDGES) for _ in range(base4)]  
	
	xLen, yLen, zLen = xRange[1] - xRange[0], yRange[1] - yRange[0], zRange[1] - zRange[0]
	for a1 in range(0, base2 + 1):
		for a2 in range(0, base2 + 1):
			for a3 in range(0, base2 + 1):
				deltaX, deltaY, deltaZ = xLen * a1 / base2, yLen * a2 / base2, zLen * a3 / base2
				x, y, z = deltaX + xRange[0], deltaY + yRange[0], deltaZ + zRange[0]
				if check1(x, y, z) or check1(z, x, y) or check1(y, z, x):
					continue
				if x > y or x > z or y > z: 
					continue
				for a4 in range(base4):
					deltaP = pLen * a4 / base4
					p0 = deltaP + plower
					p1 = p0 + 1/base4
					pl = max(0, 1- (x + y +z)/2, 1- x - y)
					pr = min(1, 1-x,1-y,1-z)
					if p1 < pl or p0 > pr:
						continue
					p0 = max(p0, pl)
					p1 = min(p1, pr)
					for p in (p0, p1):
						for i in range(len(EDGES)):
							temp = ratio(x, y, z, p, xRange, yRange, zRange, EDGES[i])
							# if f[i] < temp:
							# 	print("entry: ", x, y, z, p, xRange[0], yRange[0], zRange[0], EDGES[i], temp)
							f[a4][i] = max(temp, f[a4][i])
	
	costIdx = [np.argmax(f[i]) for i in range(base4)]

	g = [-100] * (base4 + 1)
	gcostIdx = [-100] * (base4 + 1)
	gcostIdx[0], gcostIdx[-1] = costIdx[0], costIdx[-1]
	g[0], g[-1] = f[0][gcostIdx[0]], f[-1][gcostIdx[-1]]
	for i in range(1, base):
		if f[i - 1][costIdx[i-1]] > f[i][costIdx[i]]:
			g[i], gcostIdx[i] = f[i - 1][costIdx[i-1]], costIdx[i-1]
		else:
			g[i], gcostIdx[i] = f[i][costIdx[i]], costIdx[i]		

	for x in xRange:
		for y in yRange:
			for z in zRange:
				if x > y or x > z or y > z:
					continue
				for a4 in range(base4 + 1):
					deltaP = pLen * a4 / base4
					p = deltaP + plower
					if checkDuplicate(x, y, z, p, idx, idy, idz, [True, True, True]):
						continue
					# record.add((roundIndex(x, edges[0]), roundIndex(y, edges[1]), roundIndex(z, edges[2]), roundIndex(p, True)))
					# if checkzero(x, y, z, edges) or checkzero(y, x, z, [edges[1], edges[0], edges[2]]) or \
					# 	checkzero(z, y, x, [edges[2], edges[1], edges[0]]):
					# 	continue
					covxy, covxz, covyz = covariance(z, y, x, p), \
							covariance(y, x, z, p), covariance(x, y, z, p)
					covxx, covyy, covzz = covarianceXX(x, xRange), covarianceXX(y, yRange), covarianceXX(z, zRange)
					index.append((idx, idy, idz, gcostIdx[a4], x, y, z, p, xRange[0], yRange[0], zRange[0], covxy, covxz, covyz, covxx, covyy, covzz, g[a4]))

def createtriangles1(xRange, yRange, zRange, index):
	addEdgeIndex(xRange[0], True)
	addEdgeIndex(yRange[0], True)
	addEdgeIndex(zRange[0], True)
	idx, idy, idz = edgesIndex[roundIndex(xRange[0], True)], \
	edgesIndex[roundIndex(yRange[0], True)], edgesIndex[roundIndex(zRange[0], True)]

	plower = max(0, max(1- (xRange[1] + yRange[1] +zRange[1]) / 2, 1 - (xRange[1] + yRange[1]), 1 - (xRange[1] + zRange[1]), 1 - (zRange[1] + yRange[1]) ))
	pupper = min(1, min(1 - xRange[0], 1 - yRange[0], 1 - zRange[0]))
	for x in xRange:
		for y in yRange:
			for z in zRange:
				if x > y or x > z or y > z:
					continue
				for p in [plower, pupper]:
					if checkDuplicate(x, y, z, p, idx, idy, idz, [True, True, True]):
						continue
					covxy, covxz, covyz = covariance(z, y, x, p), \
							covariance(y, x, z, p), covariance(x, y, z, p)
					covxx, covyy, covzz = covarianceXX(x, xRange), covarianceXX(y, yRange), covarianceXX(z, zRange)
					f2 = [-100] * len(EDGES)
					for i in range(len(EDGES)):
							temp = ratio1(x, y, z, p, xRange, yRange, zRange, EDGES[i])
							# if f[1] < temp:
							# 	print("entry: ", x, y, z, p, xRange[0], yRange[0], zRange[0], EDGES[i], temp)
							f2[i] = max(temp, f2[i])
					costIdx = np.argmax(f2)
					edges = EDGES[costIdx]
					index.append((idx, idy, idz, costIdx, x, y, z, p, xRange[0], yRange[0], zRange[0], covxy, covxz, covyz, covxx, covyy, covzz, f2[costIdx]))

def construct_triangles():	
	index = []

	for i in range(len(splittingPoints) - 1):
		if i % int((len(splittingPoints) - 1) / 10) == 0:
			print("progress: ", 100.0 * i / (len(splittingPoints) - 1), "%")
		for j in range(i, len(splittingPoints) - 1):
			for k in range(j, len(splittingPoints) - 1):
				x = [splittingPoints[i], splittingPoints[i + 1]]
				y = [splittingPoints[j], splittingPoints[j + 1]]
				z = [splittingPoints[k], splittingPoints[k + 1]]
				if x[1] + y[1] <= z[0]:
					break
				if x[0] >= 0.36 and x[0] <= 0.58 or y[0] >= 0.36 and y[0] <= 0.58 or z[0] >= 0.36 and z[0] <= 0.58:
					continue
				# if x[0] != 0.3 and x[0] != 0.54 and x[0] != 0.85 and x[0] != 0.99:
				# 	break
				# if x[0] >= 0.42 and x[0] <= 0.60 or x[0] >= 0.96:
				# 	1 == 1
				# else:
				# 	break
				# print(x, y, z)
				z[1] = min(x[1] + y[1], z[1])
				x[0] = max(x[0], z[0] - y[1])
				y[0] = max(y[0], z[0] - x[1])
				if (x[0] >= 0.43 and x[0] < 0.60 or x[0] >= 0.32 and x[0] <= 0.36) and (y[0] >= 0.43 and y[0] < 0.60 or y[0] >= 0.32 and y[0] <= 0.36):
					createtriangles2(x, y, z, index)
				else:
					createtriangles(x, y, z, index)
	print(len(index))
	return index

if __name__ == '__main__':
	start_time = time.time()

	splittingPoints = list(filter(filtersplittlingpoints, list(set(splittingPoints))))
	splittingPoints.sort()

	pa = pivotAnalysis()

	print(splittingPoints, len(splittingPoints))
	index = []
	index = construct_triangles()
	# # createtriangles([0, 1.0],[0.31, 0.40], [0.31, 0.40], index)
	# createtriangles([0.50, 0.50],[0.50, 0.50], [0.50, 0.50], index)
	# createtriangles([0.50, 0.50],[0.50, 0.50], [1.00, 1.00], index)
	# createtriangles([0.49, 0.50],[0.49, 0.50], [0.49, 0.50], index)
	# createtriangles([0.49, 0.50],[0.49, 0.50], [0.99, 1.00], index)
	# createtriangles1([0.472, 0.473],[0.527, 0.528], [0.995, 0.996], index)
	# # createtriangles([0.39, 0.40],[0.39, 0.40], [0.39, 0.40], index)
	# # createtriangles([0.39, 0.40],[0.59, 0.60], [0.99, 1.0], index)
	# # createtriangles([0.39, 0.40],[0.60, 0.70], [0.99, 1.0], index)
	# createtriangles([0.15, 0.20],[0.15, 0.20], [0.31, 0.38], index)
	# createtriangles([0.31, 0.38],[0.31, 0.38], [0.31, 0.38], index)
	# createtriangles1([0.31, 0.38],[0.31, 0.38], [0.31, 0.38], index)
	# createtriangles1([0.31, 0.32],[0.31, 0.32], [0.31, 0.32], index)
	# createtriangles1([0.32, 0.325],[0.32, 0.325], [0.32, 0.325], index)
	# createtriangles([0.32, 0.34],[0.32, 0.34], [0.32, 0.34], index)
	# createtriangles1([0.20, 0.28],[0.54, 0.6], [0.8, 0.88], index)
	# createtriangles([0.20, 0.28],[0.54, 0.6], [0.8, 0.88], index)
	# createtriangles([0.8, 0.9],[0.8, 0.9], [0.8, 0.9], index)
	# createtriangles([0, 0.03],[0, 0.03], [0, 0.03], index)
	# createtriangles1([0.46, 0.48],[0.46, 0.48], [0.94, 0.96], index)
	# createtriangles1([0.45, 0.46],[0.45, 0.46], [0.45, 0.46], index)
	# createtriangles1([0.45, 0.46],[0.54, 0.55], [0.99, 1.00], index)
	# createtriangles2([0.15, 0.16],[0.15, 0.16], [0.31, 0.315], index)
	# createtriangles2([0.15, 0.16],[0.15, 0.16], [0.15, 0.16], index)
	# createtriangles([0.32, 0.325],[0.32, 0.325], [0.32, 0.325], index)
	# createtriangles([0.16, 0.17],[0.16, 0.17], [0.31, 0.32], index)
	# createtriangles([0.54, 0.55],[0.54, 0.55], [0.99, 1.00], index)
	# createtriangles2([0.54, 0.55],[0.54, 0.55], [0.54, 0.55], index)
	# createtriangles([0.41, 0.42],[0.58, 0.60], [0.99, 1.0], index)
	# createtriangles([0.41, 0.42],[0.41, 0.42], [0.41, 0.42], index)
	# createtriangles([0.65, 0.70],[0.65, 0.70], [0.99, 1.00], index)
	# print(index)

	with open('triangles.csv', 'w', newline='') as f:
	    # using csv.writer method from CSV package
	    write = csv.writer(f)
	    write.writerows(index)

	print("--- %s seconds ---" % (time.time() - start_time))
