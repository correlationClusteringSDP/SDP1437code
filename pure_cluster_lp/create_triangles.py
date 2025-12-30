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

target = 1.484
pset = target / 2
ppivot = 1 - pset
alpha = target / (2 - target)

base = 20
computeRatioBase = 10
oneoverbase = 1 / base

record = set()
edgesIndex = {}
epsilon = 0.00
more_dense_range = [[0.38, 0.55, 20], [0.30, 0.65, 10]]

splittingPoints = []

splittingPoints = [0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.38, 0.39, 0.4, 0.405, 0.41, \
	0.42, 0.43, 0.44, 0.45, 0.46,0.48, 0.49,0.5, 0.51, 0.52, 0.54, 0.55, 0.57, 0.58, 0.6, 0.62, 0.65, 0.7,\
		 0.75, 0.78, 0.8, 0.9, 0.95, 0.96, 0.99, 1.0]

# splittingPoints = [0.0, 0.2, 0.4, 0.6, 0.8,1.0, 0.32, 0.34, 0.36, 0.38, 0.62, 0.64, 0.96, 0.98]

splittingPoints.append(posThreshold1)
splittingPoints.append(posThreshold2)

notInSplittingRange = []

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
			return x
			
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

# Initialize pa at module level so it's available when module is imported
# This will be used by ratio() function
pa = pivotAnalysis()

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
	return round(int((x) * 100000))
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

def getEdgeIndex(x):
	"""
	Get edge index for a point x (legacy function, kept for compatibility).
	"""
	base_idx = roundIndex(x)
	if base_idx not in edgesIndex:
		edgesIndex[base_idx] = len(edgesIndex)
	return edgesIndex[base_idx]

def getEdgeRangeIndex(valueRange, tolerance=1e-10):
	"""
	Get the index of a range in splittingPoints using binary search.
	Given a range [start, end], find the index i such that:
	splittingPoints[i] <= start and splittingPoints[i+1] >= end
	(i.e., the range is contained within the splittingPoints interval)
	
	Args:
		valueRange: A list or tuple [start, end] representing the range
		tolerance: Floating point comparison tolerance (default: 1e-10)
	
	Returns:
		The index i in splittingPoints, or -1 if no containing interval is found
	"""
	start, end = valueRange[0], valueRange[1]
	
	# Use binary search to find the position where start would be inserted
	# bisect_left returns the leftmost position where start could be inserted
	i = bisect.bisect_left(splittingPoints, start)
	
	# Find the smallest containing interval
	# We need splittingPoints[i] <= start and splittingPoints[i+1] >= end
	# Start from the position found by binary search and check adjacent positions
	for candidate_i in [i - 1, i]:
		if 0 <= candidate_i < len(splittingPoints) - 1:
			interval_start = splittingPoints[candidate_i]
			interval_end = splittingPoints[candidate_i + 1]
			# Check if the query range is contained in this interval
			if interval_start <= start + tolerance and interval_end >= end - tolerance:
				return candidate_i
	
	# Range not found in splittingPoints
	return -1

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

def createtriangles(xRange, yRange, zRange, pRange, index, debug=False):
	# Track the triangle that maximizes ratio for each edge type
	best_triangles = [None] * len(EDGES)  # Store (x, y, z, p, ratio) for each edge type
	best_ratios = [-100] * len(EDGES)
	pset = -100
	xLen, yLen, zLen = xRange[1] - xRange[0], yRange[1] - yRange[0], zRange[1] - zRange[0]
	for a1 in range(0, computeRatioBase + 1):
		for a2 in range(0, computeRatioBase + 1):
			for a3 in range(0, computeRatioBase + 1):
				deltaX, deltaY, deltaZ = xLen * a1 / computeRatioBase, yLen * a2 / computeRatioBase, zLen * a3 / computeRatioBase
				x, y, z = deltaX + xRange[0], deltaY + yRange[0], deltaZ + zRange[0]
				if check1(x, y, z) or check1(z, x, y) or check1(y, z, x):
					continue
				# Calculate pl and pu based on geometric constraints and pRange
				# NOTE: In pure_cluster_lp, we do NOT adjust pl and pu with p_min
				# We use the original pl and pu values directly, even if p is very small
				pl = max(0, 1 - x - y, 1 - y - z, 1 - x - z, pRange[0])
				pu = min(1 - x, 1 - y, 1 - z, pRange[1])
				if pl > pu:
					continue
				# Sample multiple p values in the range [pl, pu]
				# Use computeRatioBase to control sampling density
				pLen = pu - pl
				if pLen > 0:
					# Sample points including boundaries and intermediate values
					p_samples = []
					for a4 in range(0, computeRatioBase + 1):
						p_val = pl + pLen * a4 / computeRatioBase
						p_samples.append(p_val)
				else:
					# If pl == pu, just use that single value
					p_samples = [pl]
				
				for p in p_samples:
					# Use p directly without any adjustment (even if p < 1 - (x + y + z) / 2)
					# This is the key difference from pure_cluster_lp_extended_3_SA version
					for i in range(len(EDGES)):
						temp = ratio(x, y, z, p, xRange, yRange, zRange, EDGES[i])
						# Update best triangle for this edge type if we found a better ratio
						# Store the original p value (not the adjusted one) for the triangle
						if temp > best_ratios[i]:
							old_ratio = best_ratios[i]
							best_ratios[i] = temp
							best_triangles[i] = (x, y, z, p, temp)
							if debug:
								print(f"DEBUG: Ratio increased for edge_type={i} {EDGES[i]}: "
								      f"{old_ratio:.10f} -> {temp:.10f} "
								      f"(x={x:.6f}, y={y:.6f}, z={z:.6f}, p={p:.6f}, "
								      f"xRange={xRange}, yRange={yRange}, zRange={zRange}, pRange={pRange})")
						pset = max(pset, psetRatio(x, y, z, EDGES[i]))
	

	# Find the edge type with maximum ratio
	costIdx = np.argmax(best_ratios)
	if best_ratios[costIdx] == -100:
		if debug:
			print(f"DEBUG: No valid triangles found for xRange={xRange}, yRange={yRange}, zRange={zRange}, pRange={pRange}")
		return 
	
	max_ratio = best_ratios[costIdx]
	if debug:
		print(f"DEBUG: Best ratio found: {max_ratio:.10f} (edge_type={costIdx} {EDGES[costIdx]})")
		if best_triangles[costIdx] is not None:
			best_x, best_y, best_z, best_p, best_r = best_triangles[costIdx]
			print(f"DEBUG: Best triangle: x={best_x:.6f}, y={best_y:.6f}, z={best_z:.6f}, p={best_p:.6f}, ratio={best_r:.10f}")
	
	# Calculate plower and pupper for boundary triangles
	plower = max(0, 1 - (xRange[1] + yRange[1]), 1 - (xRange[1] + zRange[1]), \
		1 - (zRange[1] + yRange[1]), pRange[0])
	pupper = min(1 - xRange[0], 1 - yRange[0], 1 - zRange[0], pRange[1])
	
	idx = getEdgeRangeIndex(xRange)
	idy = getEdgeRangeIndex(yRange)
	idz = getEdgeRangeIndex(zRange)
	# Add all triangles at the boundaries
	for x in xRange:
		for y in yRange:
			for z in zRange:
				if x > y or x > z or y > z:
					continue
				for p in [plower, pupper]:
					covxy, covxz, covyz = covariance(z, y, x, p), \
						covariance(y, x, z, p), covariance(x, y, z, p)
					
					# Get indices considering boundary markers
					# Check if each point is at left or right boundary of its range
					
					if debug:
						print(f"DEBUG: Saving boundary triangle: x={x}, y={y}, z={z}, p={p:.6f}, max_ratio={max_ratio:.10f}, costIdx={costIdx}")
					
					index.append(list([idx, idy, idz, x, y, z, p, max_ratio, costIdx, -1111] + \
					xRange + yRange + zRange + pRange + [ -2222, covxy, covxz, covyz, pset] ))

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
	"""
	Filter triangles by range index, edge index, and p value.
	Structure: dup_triangles[(range_idx_x, range_idx_y, range_idx_z)][(edge_idx_x, edge_idx_y, edge_idx_z)][idp] = item
	For the same (range, edge, p) combination, keep the one with maximum ratio.
	"""
	dup_triangles = {}
	ret = []
	ratio_idx = 7  # max_ratio is at index 7

	for item in index:
		# New format: [idx, idy, idz, x, y, z, p, max_ratio, costIdx, -1111, ...]
		# idx, idy, idz are range indices (from getEdgeRangeIndex)
		range_idx_x, range_idx_y, range_idx_z = item[0], item[1], item[2]
		x, y, z, p = item[3], item[4], item[5], item[6]
	
		# Get edge indices using getEdgeIndex for x, y, z values
		edge_idx_x = getEdgeIndex(x)
		edge_idx_y = getEdgeIndex(y)
		edge_idx_z = getEdgeIndex(z)
		
		# Get p index
		idp = roundIndex(p)
		
		# Create nested structure: range_index -> edge_index -> p_index
		range_key = (range_idx_x, range_idx_y, range_idx_z)
		edge_key = (edge_idx_x, edge_idx_y, edge_idx_z)
		
		if range_key not in dup_triangles:
			dup_triangles[range_key] = {}
		if edge_key not in dup_triangles[range_key]:
			dup_triangles[range_key][edge_key] = {}
		
		# For the same (range, edge, p) combination, keep the one with maximum ratio
		if idp not in dup_triangles[range_key][edge_key]:
			dup_triangles[range_key][edge_key][idp] = item
		elif len(item) > ratio_idx and len(dup_triangles[range_key][edge_key][idp]) > ratio_idx:
			if dup_triangles[range_key][edge_key][idp][ratio_idx] < item[ratio_idx]:
				dup_triangles[range_key][edge_key][idp] = item

	print("before removing duplicated p range triangles, ", 
		  sum(sum(len(inner) for inner in edge_dict.values()) for edge_dict in dup_triangles.values()))
	
	# Process each range -> edge -> p combination
	for range_key in dup_triangles.keys():
		for edge_key in dup_triangles[range_key].keys():
			pairs_with_idp = dup_triangles[range_key][edge_key]
			# Convert from idp keys to actual p value keys for upper_base_from_arraydict
			pairs = {}
			for idp, item in pairs_with_idp.items():
				# New format: [idx, idy, idz, x, y, z, p, max_ratio, ...]
				actual_p = item[6]  # p is at index 6
				pairs[actual_p] = item
			
			filtered_prange_triangles = upper_base_from_arraydict(pairs, value_index=ratio_idx, drop_collinear=True)
			for key, value in filtered_prange_triangles.items():
				ret.append(value)

	return ret

if __name__ == '__main__':
	start_time = time.time()

	splittingPoints = list(filter(filtersplittlingpoints, list(set(splittingPoints))))
	splittingPoints.sort()

	print(splittingPoints, len(splittingPoints))
	index = []
	index = construct_triangles()

	trianlges = filter_index(index)

	print("all triangles:", len(index))
	print("after filter:", len(trianlges))

	with open('triangles_merged.csv', 'w', newline='') as f:
	    # using csv.writer method from CSV package
		write = csv.writer(f)
		write.writerows(trianlges)

	print("edgegIndex: ", edgesIndex)

	print("--- %s seconds ---" % (time.time() - start_time))
