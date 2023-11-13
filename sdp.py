import cvxpy as cp
import numpy as np
import csv
from collections import defaultdict
from create_triangles import *

def solve_sdp(triangles):
	# n = 1002
	scale = 500

	l = len(triangles)
	
	ones = np.ones(l)
	c = np.array([triangles[i][17] for i in range(l)])
	allvalues = list(set([triangles[i][0] for i in range(l)] + [triangles[i][1] for i in range(l)] + [triangles[i][2] for i in range(l)]))
	numOfEdges = max(allvalues) + 1

	# index = {}
	# count = 0
	# for value in allvalues:
	# 	index[value] = count
	# 	count += 1
	# print(np.shape(ones))
	# print(np.shape(c))
	# print(index, len(index))
		 
	# Define and solve the CVXPY problem.
	x = cp.Variable(l)
	maxValue = cp.Variable()

#####################################################
##covariance constraint
#####################################################
	B = cp.Variable((numOfEdges, numOfEdges))

	C = np.empty((numOfEdges,numOfEdges),dtype=object)

	for i in range(numOfEdges):
		for j in range(numOfEdges):
			C[i ,j] = defaultdict(int)
	for i in range(l):
		idx, idy, idz = triangles[i][0], triangles[i][1], triangles[i][2]
		tx, ty, tz = triangles[i][4], triangles[i][5], triangles[i][6]
		covxy, covxz, covyz = triangles[i][11], triangles[i][12], triangles[i][13]
		covxx, covyy, covzz = triangles[i][14], triangles[i][15], triangles[i][16]
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
	constraints = []
	constraints += [B >> 0]
	constraints += [ones.T @ x <= scale]
	constraints += [x >= 0]
	for i in range(numOfEdges):
		for j in range(numOfEdges):
			if len(C[i,j]) > 0:
				# if i == 5 and j == 5:e
				# 	print([x[v[0]]*v[1] for v in C[i, j]])
				constraints += [B[i,j] == cp.sum([coefficient*x[idx] for idx, coefficient in C[i, j].items()])]
			else:
				constraints += [B[i,j] == 0]

###########################################################
###frequency constraint
###########################################################

	F = cp.Variable((numOfEdges, numOfEdges))

	FC = np.empty((numOfEdges,numOfEdges),dtype=object)
	for i in range(numOfEdges):
		for j in range(numOfEdges):
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

	constraints += [F >> 0]

	for i in range(numOfEdges):
		for j in range(numOfEdges):
			if len(FC[i,j]) > 0:
				# if i == 5 and j == 5:
				# 	print([x[v[0]]*v[1] for v in C[i, j]])
				constraints += [F[i,j] == cp.sum([coefficient * x[idx] for idx, coefficient in FC[i, j].items()])]
			else:
				constraints += [F[i,j] == 0]


###########################################################
###objective
###########################################################

	prob = cp.Problem(cp.Maximize(c.T @ x), constraints)

	
	prob.solve()
	
	# prob.solve(solver=cp.CVXOPT)
	# prob.solve(solver=cp.SCS)
	# prob.solve(solver=cp.MOSEK)

	return x, prob

if __name__ == '__main__':
	start_time = time.time()
	file = open("triangles.csv", "r")
	alltriangles = [list(map(int,rec[0:4])) + list(map(float,rec[4:])) for rec in csv.reader(file, delimiter=',')]
	file.close()

	# print(alltriangles)
	print("solving sdp")
	x, prob = solve_sdp(alltriangles)
	output_file = 'output.txt'
	# Print result.
	print("\ntarget ratio: ", target, ", base: ", base, file=open(output_file, 'a'))
	print("\nThe optimal value is", prob.value, file=open(output_file, 'a'))
	print("A solution x of x >= 1 is", file=open(output_file, 'a'))

	l = len(x.value)
	count = 0
	total = 0
	xResults = []
	last = []
	for i in range(l):
		if x.value[i] > 0.01:
			xResults.append((x.value[i], i))
			last.append(tuple(alltriangles[i]))
	xResults.sort(reverse = True)
	for i in range(len(xResults)):
		value, idx = xResults[i]
		if alltriangles[idx][17] != 0:
			print(idx, alltriangles[idx], value, file=open(output_file, 'a'))
	for i in range(min(len(xResults),20)):
		value, idx = xResults[i]
		if alltriangles[idx][17] == 0:
			print(idx, alltriangles[idx], value, file=open(output_file, 'a'))

	print('\n', file=open('last.csv', 'a'))
	with open('last.csv', 'a', newline='') as f:
	    # using csv.writer method from CSV package
	    write = csv.writer(f)
	    write.writerows(last)

	print("--- %s seconds ---" % (time.time() - start_time), file=open(output_file, 'a'))