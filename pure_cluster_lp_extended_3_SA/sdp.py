from typing import Any

import sys
import os
import cvxpy as cp
import numpy as np
import mosek
import inspect
from scipy.sparse import coo_matrix
import csv
from collections import defaultdict
from create_triangles import *


# n = 502
scale = 1000000

def solve_sdp(triangles):
	l = len(triangles)
	
	ones = np.ones(l)
	c = np.array([triangles[i][7] for i in range(l)])
	allvalues = list[Any](set([triangles[i][0] for i in range(l)] + [triangles[i][1] for i in range(l)] + [triangles[i][2] for i in range(l)]))
	numofEdgeRangeTypes = max(allvalues) + 1
		 
	# Define and solve the CVXPY problem.
	x = cp.Variable(l)
	# m = l (number of triangles)

#####################################################
## constraint
#####################################################

	constraints = []

	constraints += [ones.T @ x <= scale]
	constraints += [x >= 0]

	# pset = np.array([triangles[i][25] for i in range(l)])
	# constraints += [pset.T @ x >= 0]

#####################################################
##covariance constraint
#####################################################
	C = np.empty((numofEdgeRangeTypes, numofEdgeRangeTypes),dtype=object)

	for i in range(numofEdgeRangeTypes):
		for j in range(numofEdgeRangeTypes):
			C[i ,j] = defaultdict(int)
	for i in range(l):
		idx, idy, idz = triangles[i][0], triangles[i][1], triangles[i][2]
		# New format: covxy, covxz, covyz are at indices 19, 20, 21
		covxy, covxz, covyz = triangles[i][19], triangles[i][20], triangles[i][21]
		C[idx, idy][i] +=  covxy
		C[idy, idx][i] +=  covxy
		C[idx, idz][i] +=  covxz
		C[idz, idx][i] +=  covxz
		C[idy, idz][i] +=  covyz
		C[idz, idy][i] +=  covyz

	def row_id(i, j):
		return i * numofEdgeRangeTypes + j  # flatten (i,j) in row-major order

	rows, cols, data = [], [], []
	for i in range(numofEdgeRangeTypes):
		for j in range(numofEdgeRangeTypes):
			d = C[i, j]
			if not d:
				continue
			r = row_id(i, j)
			for idx, coef in d.items():
				rows.append(r)
				cols.append(int(idx))
				data.append(float(coef))

	# Build sparse map S so vec(B) = S @ x
	S = coo_matrix((data, (rows, cols)), shape=(numofEdgeRangeTypes * numofEdgeRangeTypes, l))

	# Affine expression for B, then symmetrize
	B_vec = S @ x
	B_raw = cp.reshape(B_vec, (numofEdgeRangeTypes, numofEdgeRangeTypes), order='C') 

	constraints += [B_raw >> 0]

###########################################################
###frequency constraint
###########################################################
	FC = np.empty((numofEdgeRangeTypes,numofEdgeRangeTypes),dtype=object)

	for i in range(numofEdgeRangeTypes):
		for j in range(numofEdgeRangeTypes):
			FC[i ,j] = defaultdict(int)
	for i in range(l):
		idx, idy, idz = triangles[i][0], triangles[i][1], triangles[i][2]
		FC[idx, idy][i] +=  1
		FC[idy, idx][i] +=  1
		FC[idx, idz][i] +=  1
		FC[idz, idx][i] +=  1
		FC[idy, idz][i] +=  1
		FC[idz, idy][i] +=  1
		# FC[idx, idx][i] +=  2 / (n - 2)
		# FC[idy, idy][i] +=  2 / (n - 2)
		# FC[idz, idz][i] +=  2 / (n - 2)

	rows, cols, data = [], [], []
	for i in range(numofEdgeRangeTypes):
		for j in range(numofEdgeRangeTypes):
			d = FC[i, j]
			if not d:
				continue
			r = row_id(i, j)
			for idx, coef in d.items():
				rows.append(r)
				cols.append(int(idx))
				data.append(float(coef))

	# Build sparse map A so vec(B) = A @ x
	S = coo_matrix((data, (rows, cols)), shape=(numofEdgeRangeTypes * numofEdgeRangeTypes, l))

	# Affine expression for B, then symmetrize
	F_vec = S @ x
	F_raw = cp.reshape(F_vec, (numofEdgeRangeTypes, numofEdgeRangeTypes), order='C') 

	constraints += [F_raw >> 0]

###########################################################
###objective
###########################################################

	prob = cp.Problem(cp.Maximize(c.T @ x), constraints)

	# prob.solve()
	
	prob.solve(solver=cp.SCS, eps=1e-5, verbose=True)	
	# prob.solve(solver=cp.CVXOPT)
	# prob.solve(solver=cp.SCS)
	# prob.solve(solver=cp.MOSEK, eps=1e-5, verbose=True)

	return x, prob

if __name__ == '__main__':
	# Parse command line arguments
	# Default values assume running from the subdirectory (local execution)
	# Server execution always passes explicit paths
	triangles_file = sys.argv[1] if len(sys.argv) > 1 else "triangles_merged.csv"
	output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
	
	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)
	
	splittingPoints = list(filter(filtersplittlingpoints, list(set(splittingPoints))))
	splittingPoints.sort()

	start_time = time.time()
	file = open(triangles_file, "r")
	alltriangles = [list(map(int,rec[0:3])) + list(map(float,rec[3:])) for rec in csv.reader(file, delimiter=',')]
	file.close()

	# print(alltriangles)
	print("solving sdp")
	print(f"Number of triangles: {len(alltriangles):,}")
	
	# Optional: Monitor memory usage (uncomment if psutil is installed)
	# try:
	#     import psutil
	#     process = psutil.Process(os.getpid())
	#     mem_before = process.memory_info().rss / 1024 / 1024  # MB
	#     print(f"Memory before solving: {mem_before:.2f} MB")
	# except ImportError:
	#     pass
	
	x, prob = solve_sdp(alltriangles)
	
	# Optional: Monitor memory usage after solving
	# try:
	#     import psutil
	#     process = psutil.Process(os.getpid())
	#     mem_after = process.memory_info().rss / 1024 / 1024  # MB
	#     print(f"Memory after solving: {mem_after:.2f} MB")
	#     print(f"Memory increase: {mem_after - mem_before:.2f} MB")
	# except ImportError:
	#     pass
	output_file = os.path.join(output_dir, 'output.txt')
	# Print result.
	print("\ntarget ratio: ", target, ", base: ", base, file=open(output_file, 'a'))
	print("\nconfiguration, posThreshold1: ", posThreshold1, " posThreshold2: ", posThreshold2, file=open(output_file, 'a'))

	print("\n", inspect.getsource(pivotAnalysis.probCorrelatedInput), file=open(output_file, 'a'))

	print("\nsplitting point:, ", splittingPoints, file=open(output_file, 'a'))
	print("\nThe optimal value is", prob.value, file=open(output_file, 'a'))
	print("A solution x of x >= 1 is", file=open(output_file, 'a'))

	l = len(x.value)
	count = 0
	total = 0
	xResults = []
	last = []
	for i in range(l):
		if x.value[i] > scale / 5000.0:
			xResults.append((x.value[i], i))
			last.append(tuple(alltriangles[i]))
	xResults.sort(reverse = True)
	for i in range(len(xResults)):
		value, idx = xResults[i]
		# New format: max_ratio is at index 7
		if alltriangles[idx][7] != 0:
			print(idx, alltriangles[idx], value, file=open(output_file, 'a'))
	for i in range(min(len(xResults),20)):
		value, idx = xResults[i]
		# New format: max_ratio is at index 7
		if alltriangles[idx][7] == 0:
			print(idx, alltriangles[idx], value, file=open(output_file, 'a'))

	last_csv_file = os.path.join(output_dir, 'last.csv')
	print('\n', file=open(last_csv_file, 'a'))
	with open(last_csv_file, 'a', newline='') as f:
	    # using csv.writer method from CSV package
		write = csv.writer(f)
		write.writerows(last)

	print("--- %s seconds ---" % (time.time() - start_time), file=open(output_file, 'a'))