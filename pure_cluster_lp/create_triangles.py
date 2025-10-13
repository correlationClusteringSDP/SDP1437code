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

posThreshold1 = 0.40
posThreshold2 = 0.57
# negThreshold1 = -1
# negThreshold2 = -1

EDGES = [[True, True, True], [True, True, False], [True, False, True], \
[False, True,  True], [True, False, False], [False, True, False],[False, False , True], [False, False, False]]

target = 1.485
pset = target / 2
ppivot = 1 - pset
alpha = target / (2 - target)

base = 20
computeRatioBase = 10
splittingPointsBase = 10
oneoverbase = 1 / base
record = set()
edgesIndex = {}
splittingPoints = []
epsilon = 0.00
more_dense_range = [[0.38, 0.45, 20], [0.37, 0.65, 10]]

splittingPoints = []
splittingPoints.append(posThreshold1)
splittingPoints.append(posThreshold2)
# splittingPoints = list(reduce(lambda x, y: x + y, Threshold))


# splittingPoints += [0.39, 0.4, 0.41, 0.42]
splittingPoints += [i / base for i in range(0, base + 1)]
splittingPoints += [0.405, 0.58, 0.78, 0.96, 0.99]
splittingPoints += [i / 100 for i in range(38, 45, 1)]

# splittingPoints += [0.05, 0.95, 0.51, 0.49, 0.99]/
# splittingPoints += [i / 100 for i in range(45, 60, 2)]
# splittingPoints += [0.45 + 0.1 * i / splittingPointsBase for i in range(0, splittingPointsBase, 1)]
# splittingPoints += [i / 100 for i in range(38, 56, 1)]
# splittingPoints += [i / 100 for i in range(94, 100, 2)]
# splittingPoints += [0.95 + 0.1 * i / splittingPointsBase for i in range(0, splittingPointsBase, 1)]

notInSplittingRange = [0.15, 0.43, 0.45, 0.25, 0.62, 0.85]

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

	def correlatedRounding(self, xRange, positive):
		xleft, xright = xRange
		if positive:
			if xright > self.posCorrelatedRoundingThreshold1 and xright <= self.posCorrelatedRoundingThreshold2:
				return True
			else:
				return False
		else:
			return False

	def probCorrelatedInput(self, x, xRange, positive):
		xleft, xright = xRange
		if positive:
			if xright <= self.posThreshold1:
				return 0
			elif xright <= self.posThreshold2:
				return x
			else:
				return x
				# return (x - self.posThreshold2) * (x - self.posThreshold2) / (1 - self.posThreshold2) + self.posThreshold2
		else:
			return x * x
			
	def edgealgCorrelatedInput(self, x, y, z, p, yRange, zRange, edges):
		pvw = self.probCorrelatedInput(y, yRange, edges[1])
		puw = self.probCorrelatedInput(z, zRange, edges[2])

		# print("edgealg:", yinput, zinput, self.correlatedRounding(yinput, edges[1]), self.correlatedRounding(zinput, edges[2]))
		if self.correlatedRounding(yRange, edges[1]) and self.correlatedRounding(zRange, edges[2]):
			# print(x, y, z, p, edges, "correlatedRounding",  yinput, zinput)
			if edges[0]:
				return ((1 - y) + (1 - z) - 2 * p)
			else:
				return p
		else:
			if edges[0]:
				return puw + pvw - 2 * puw * pvw
			else:
				return (1 - puw) * (1 - pvw)

	def edgelpCorrelatedInput(self, x, y, z, p, yRange, zRange, edges):
		pvw = self.probCorrelatedInput(y, yRange, edges[1])
		puw = self.probCorrelatedInput(z, zRange, edges[2])

		if self.correlatedRounding(yRange, edges[1]) and self.correlatedRounding(zRange, edges[2]):
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
	if x in notInSplittingRange:
		return False

	for left, right in interval:
		if x >= left and x <= right:
			return True
	return False

def roundIndex(x):
	return round(int((x) * 10000))
	# return round(int(x * 10000))

def psetEdgeAlgCost(x, positive):
	if positive:
		return 2 * x / (1 + x) 
	else:
		return (1 - x) / (1 + x)

def psetEdgelp(x, positive):
	if positive:
		return x
	else:
		return 1 - x

def psetRatio(x, y, z, edges):
	s1 = psetEdgeAlgCost(x, edges[0]) + psetEdgeAlgCost(y, edges[1]) + psetEdgeAlgCost(z, edges[2])
	s2 = psetEdgelp(x, edges[0]) + psetEdgelp(y, edges[1]) + psetEdgelp(z, edges[2])
	# return s1 - s2 + errorForSet
	return s1 - target * s2

def coefficent(x, positive):
	if positive:
		return 2 * alpha  * (x / (1 + x) )
	else:
		return alpha * (1 + 2*x) / (1 + x)

def ratio(x, y, z, p, xRange, yRange, zRange, edges):
	s1 = pa.edgealgCorrelatedInput(x, y, z, p, yRange, zRange, edges) + \
	pa.edgealgCorrelatedInput(z, y, x, p, yRange, xRange, [edges[2], edges[1], edges[0]]) + \
	pa.edgealgCorrelatedInput(y, x, z, p, xRange, zRange, [edges[1], edges[0], edges[2]])

	s2 = coefficent(x, edges[0]) * pa.edgelpCorrelatedInput(x, y, z, p, yRange, zRange, edges) + \
	coefficent(z, edges[2])*  pa.edgelpCorrelatedInput(z, y, x, p, yRange, xRange, [edges[2], edges[1], edges[0]]) + \
	coefficent(y, edges[1]) * pa.edgelpCorrelatedInput(y, x, z, p, xRange, zRange, [edges[1], edges[0], edges[2]])

	return s1 - s2 

def check1(a, b, c):
	return a + b < c

def check2(a, b, c, p):
	return a + b + c + 2 * p < 2 or a + b + p < 1 or a + c + p < 1 or b + c + p < 1 or a + p > 1 or b + p > 1 or c + p >1

def check3(a, b, c):
	return 0.5 - ita <= a <= 0.5 + ita and 0.5 - ita <= b <= 0.5 + ita and ( c <= 0.5 + 5*ita or c >= 1 - ita)

def addEdgeIndex(x):
	idx = roundIndex(x)
	if idx not in edgesIndex:
		edgesIndex[idx] = len(edgesIndex)

def getEdgeIndex(x):
	idx = roundIndex(x)
	if idx not in edgesIndex:
		edgesIndex[idx] = len(edgesIndex)
	return edgesIndex[idx]

def covariance(x, y, z, p):
	return p - (1 - y)*(1 - z)

def covarianceXX(x):
	return x - x * x

def createPRange(xRange, yRange, zRange):
	in_dense_range = -1
	for i in range(len(more_dense_range)):
		r = more_dense_range[i]
		if xRange[0] > r[0] and xRange[0] <= r[1] and yRange[0] > r[0] and yRange[0] <= r[1]:
			in_dense_range = i
			break

	ret = []
	plower = max(0, 1 - (xRange[1] + yRange[1]), 1 - (xRange[1] + zRange[1]), \
		1 - (zRange[1] + yRange[1]))
	pupper = min(1 - xRange[0], 1 - yRange[0], 1 - zRange[0])
	pLen = pupper - plower

	if pLen < 0:
		return []

	if in_dense_range < 0:
		ret.append([plower, pupper])
	else:
		pbase = more_dense_range[in_dense_range][2]
		for a4 in range(pbase):
			p1 = plower + pLen * a4 / pbase
			p2 = p1 + pLen / pbase
			ret.append([p1, p2])


	return ret

def createtriangles(xRange, yRange, zRange, pRange, index):
	idx, idy, idz = getEdgeIndex(xRange[0]), getEdgeIndex(yRange[0]), getEdgeIndex(zRange[0])

	f = [-100] * len(EDGES)
	pset = -100
	xLen, yLen, zLen = xRange[1] - xRange[0], yRange[1] - yRange[0], zRange[1] - zRange[0]
	for a1 in range(0, computeRatioBase + 1):
		for a2 in range(0, computeRatioBase + 1):
			for a3 in range(0, computeRatioBase + 1):
				deltaX, deltaY, deltaZ = xLen * a1 / computeRatioBase, yLen * a2 / computeRatioBase, zLen * a3 / computeRatioBase
				x, y, z = deltaX + xRange[0], deltaY + yRange[0], deltaZ + zRange[0]
				if check1(x, y, z) or check1(z, x, y) or check1(y, z, x):
					continue
				pl = max(0, 1 - x - y, 1 - y - z, 1- x - z, pRange[0])
				pu = min(1 - x, 1 - y, 1- z, pRange[1])
				if pl > pu:
					continue
				for p in [pl, pu]:
					for i in range(len(EDGES)):
						temp = ratio(x, y, z, p, xRange, yRange, zRange, EDGES[i])
						# if temp == 0.15188976411630217:
						# 	print(x, y, z, p, xRange, yRange, zRange, pRange)
						# if f[i] < temp:
						# 	print("entry: ", x, y, z, p, xRange[0], yRange[0], zRange[0], EDGES[i], temp)
						f[i] = max(temp, f[i])
						pset = max(pset, psetRatio(x, y, z, EDGES[i]))
						

	costIdx = np.argmax(f)
	edges = EDGES[costIdx]
	plower = max(0, 1 - (xRange[1] + yRange[1]), 1 - (xRange[1] + zRange[1]), \
		1 - (zRange[1] + yRange[1]), pRange[0])
	pupper = min(1 - xRange[0], 1 - yRange[0], 1 - zRange[0], pRange[1])
	for x in xRange:
		for y in yRange:
			for z in zRange:
				if x > y or x > z or y > z:
					continue
				for p in [plower, pupper]:
					covxy, covxz, covyz = covariance(z, y, x, p), \
							covariance(y, x, z, p), covariance(x, y, z, p)
					covxx, covyy, covzz = covarianceXX(x), covarianceXX(y), covarianceXX(z)
					index.append(list([x, y, z, p, covxy, covxz, covyz, covxx, covyy, \
						covzz, f[costIdx], costIdx, -1111] + xRange + yRange + zRange +\
						pRange + [-2222] + [pset]))

def construct_triangles():	
	index = []
		
	for i in range(len(splittingPoints) - 1):
		if i % int(max((len(splittingPoints) - 1) / 10, 1)) == 0:
			print("progress: ", 100.0 * i / (len(splittingPoints) - 1), "%")
		for j in range(i, len(splittingPoints) - 1):
			for k in range(j, len(splittingPoints) - 1):
				xRange = [splittingPoints[i], splittingPoints[i + 1]]
				yRange = [splittingPoints[j], splittingPoints[j + 1]]
				zRange = [splittingPoints[k], splittingPoints[k + 1]]
				if xRange[1] + yRange[1] <= zRange[0]:
					break
				zRange[1] = min(xRange[1] + yRange[1], zRange[1])
				xRange[0] = max(xRange[0], zRange[0] - yRange[1])
				yRange[0] = max(yRange[0], zRange[0] - xRange[1])
				pRanges = createPRange(xRange, yRange, zRange)
				if len(pRanges) == 0:
					continue
				for pRange in pRanges:
					createtriangles(xRange, yRange, zRange, pRange, index)
		
	return index

def upper_base_from_dict(d, tol=1e-12):
    best = {}
    for x, y in d.items():
        if x not in best or y > best[x]:
            best[x] = y

    # 1) sort by x
    pts = sorted(best.items(), key=lambda t: t[0])  # list of (x, y)

    # 2) helper to check if middle point b is strictly below line a--c
    def below_chord(a, b, c):
        (x1, y1), (x2, y2), (x3, y3) = a, b, c
        if abs(x3 - x1) <= tol:
            # x1 == x3 (vertical line) -> cannot define alpha; keep b
            return False
        alpha = (x3 - x2) / (x3 - x1)  # so x2 = alpha*x1 + (1-alpha)*x3
        y_on = alpha * y1 + (1 - alpha) * y2
        return (y2 < y_on - tol)  # strict inequality (with tolerance)

    # 3) monotone chain for the *upper* envelope
    hull = []
    for p in pts:
        while len(hull) >= 2 and below_chord(hull[-2], hull[-1], p):
            hull.pop()
        hull.append(p)

    # hull now contains only the base points (in increasing x)
    return dict(hull)

def upper_base_from_arraydict(d, value_index = 10, drop_collinear=False, tol=1e-12):
	pts = []
	for x, arr in d.items():
		if len(arr) <= value_index:
			raise IndexError(f"x={x} has array of length {len(arr)} < value_index+1")
		y = float(arr[value_index])
		pts.append((x, y))
	pts.sort(key=lambda t: t[0])

	if len(pts) <= 2:
		return {x: d[x] for x, _ in pts}

	hull = []
	for p in pts:
		while len(hull) >= 2:
			(x1, y1) = hull[-2]
			(x2, y2) = hull[-1]
			(x3, y3) = p
			lhs = (y2 - y1) * (x3 - x1)
			rhs = (y3 - y1) * (x2 - x1)

			if drop_collinear:
				cond = lhs <= rhs + tol   # remove collinear middles too
			else:
				cond = lhs < rhs - tol    # keep collinear middles

			if cond:
				hull.pop()
			else:
				break
		hull.append(p)

	kept_keys = {x for x, _ in hull}
	keybase = {x: d[x] for x in kept_keys}

	return keybase

def filter_index(index):
	dup_triangles = {}
	ret = []

	for item in index:
		x, y, z, p = item[0:4]
		# print(x, y, z, p)
		idx, idy, idz, idp = getEdgeIndex(x), getEdgeIndex(y), getEdgeIndex(z), roundIndex(p) 
		if (idx, idy, idz) not in dup_triangles:
			dup_triangles[(idx, idy, idz)] = {}
			dup_triangles[(idx, idy, idz)][p] = item
		elif p not in dup_triangles[(idx, idy, idz)]:
			dup_triangles[(idx, idy, idz)][p] = item	
		elif dup_triangles[(idx, idy, idz)][p][10] < item[10]:
			dup_triangles[(idx, idy, idz)][p] = item
		# if x == 0.4 and y == 0.4 and z == 0.4:
		# 	print(p, dup_triangles[(idx, idy, idz)][p])
		# if item[17] > 0:
		# 	print(item)
	# print(dup_triangles)

	print("before removing duplicated p range triangles, ", sum(len(inner) for inner in dup_triangles.values()))
	for idxyz in dup_triangles.keys():
		pairs = dup_triangles[idxyz]
		# print(idxyz, pairs)
		# filtered_prange_triangles = pairs
		filtered_prange_triangles = upper_base_from_arraydict(pairs)
		# print(filtered_prange_triangles)
		for key, value in filtered_prange_triangles.items():
			ret.append(list(idxyz) + value)

	return ret

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

	trianlges = filter_index(index)

	print("all triangles:", len(index))
	print("after filter:", len(trianlges))

	with open('triangles.csv', 'w', newline='') as f:
	    # using csv.writer method from CSV package
	    write = csv.writer(f)
	    write.writerows(trianlges)

	print("edgegIndex: ", edgesIndex)

	print("--- %s seconds ---" % (time.time() - start_time))
