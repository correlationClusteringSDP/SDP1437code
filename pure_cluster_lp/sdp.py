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
	c = np.array([triangles[i][13] for i in range(l)])
	allvalues = list(set([triangles[i][0] for i in range(l)] + [triangles[i][1] for i in range(l)] + [triangles[i][2] for i in range(l)]))
	numOfEdgesTypes = max(allvalues) + 1
		 
	# Define and solve the CVXPY problem.
	x = cp.Variable(l)
	maxValue = cp.Variable()
	m = int(x.shape[0])

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
	C = np.empty((numOfEdgesTypes, numOfEdgesTypes),dtype=object)

	for i in range(numOfEdgesTypes):
		for j in range(numOfEdgesTypes):
			C[i ,j] = defaultdict(int)
	for i in range(l):
		idx, idy, idz = triangles[i][0], triangles[i][1], triangles[i][2]
		covxy, covxz, covyz = triangles[i][7], triangles[i][8], triangles[i][9]
		covxx, covyy, covzz = triangles[i][10], triangles[i][11], triangles[i][12]
		C[idx, idy][i] +=  covxy
		C[idy, idx][i] +=  covxy
		C[idx, idz][i] +=  covxz
		C[idz, idx][i] +=  covxz
		C[idy, idz][i] +=  covyz
		C[idz, idy][i] +=  covyz
		# C[idx, idx][i] +=  2 * covxx / (n - 2)
		# C[idy, idy][i] +=  2 * covyy / (n - 2)
		# C[idz, idz][i] +=  2 * covzz / (n - 2)
		# C[idz, idz].append((i, (tz - tz*tz)/n))


	# print(C[5,5])
	# print(x)


	def row_id(i, j):
		return i * numOfEdgesTypes + j  # flatten (i,j) in row-major order

	rows, cols, data = [], [], []
	for i in range(numOfEdgesTypes):
		for j in range(numOfEdgesTypes):
			d = C[i, j]
			if not d:
				continue
			r = row_id(i, j)
			for idx, coef in d.items():
				rows.append(r)
				cols.append(int(idx))
				data.append(float(coef))

	# Build sparse map S so vec(B) = S @ x
	S = coo_matrix((data, (rows, cols)), shape=(numOfEdgesTypes * numOfEdgesTypes, m))

	# Affine expression for B, then symmetrize
	B_vec = S @ x
	B_raw = cp.reshape(B_vec, (numOfEdgesTypes, numOfEdgesTypes), order='C') 

	constraints += [B_raw >> 0]

###########################################################
###frequency constraint
###########################################################
	FC = np.empty((numOfEdgesTypes,numOfEdgesTypes),dtype=object)

	for i in range(numOfEdgesTypes):
		for j in range(numOfEdgesTypes):
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
	for i in range(numOfEdgesTypes):
		for j in range(numOfEdgesTypes):
			d = FC[i, j]
			if not d:
				continue
			r = row_id(i, j)
			for idx, coef in d.items():
				rows.append(r)
				cols.append(int(idx))
				data.append(float(coef))

	# Build sparse map A so vec(B) = A @ x
	S = coo_matrix((data, (rows, cols)), shape=(numOfEdgesTypes * numOfEdgesTypes, m))

	# Affine expression for B, then symmetrize
	F_vec = S @ x
	F_raw = cp.reshape(F_vec, (numOfEdgesTypes, numOfEdgesTypes), order='C') 

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
	splittingPoints = list(filter(filtersplittlingpoints, list(set(splittingPoints))))
	splittingPoints.sort()

	start_time = time.time()
	file = open("triangles.csv", "r")
	alltriangles = [list(map(int,rec[0:3])) + list(map(float,rec[3:])) for rec in csv.reader(file, delimiter=',')]
	file.close()

	# print(alltriangles)
	print("solving sdp")
	x, prob = solve_sdp(alltriangles)
	output_file = 'output.txt'
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
		if alltriangles[idx][13] != 0:
			print(idx, alltriangles[idx], value, file=open(output_file, 'a'))
	for i in range(min(len(xResults),20)):
		value, idx = xResults[i]
		if alltriangles[idx][13] == 0:
			print(idx, alltriangles[idx], value, file=open(output_file, 'a'))

	print('\n', file=open('last.csv', 'a'))
	with open('last.csv', 'a', newline='') as f:
	    # using csv.writer method from CSV package
	    write = csv.writer(f)
	    write.writerows(last)

	print("--- %s seconds ---" % (time.time() - start_time), file=open(output_file, 'a'))