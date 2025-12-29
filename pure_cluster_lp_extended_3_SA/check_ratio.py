import math
import csv
from collections import defaultdict
import numpy as np
import time
import itertools
import sys
import os

# Import from create_triangles module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_triangles import (
    createtriangles, createPRange, EDGES, 
    check1, ratio, covariance,
    splittingPoints, filtersplittlingpoints,
    target, base, alpha  # Import constants
)

def check_ratio_for_range(xRange, yRange, zRange, pRanges=None, verbose=True):
	"""
	Call createtriangles for a range and find the maximum ratio triangle.
	This function uses the createtriangles function from create_triangles.py.
	
	Parameters:
		xRange: [x_min, x_max]
		yRange: [y_min, y_max]
		zRange: [z_min, z_max]
		pRanges: List of p ranges. If None, will compute using createPRange
		verbose: If True, print detailed information
	
	Returns:
		dict: Dictionary containing maximum ratio and triangle information
	"""
	if verbose:
		print("="*80)
		print(f"Checking ratio for range:")
		print(f"  xRange: {xRange}")
		print(f"  yRange: {yRange}")
		print(f"  zRange: {zRange}")
		print("="*80)
	
	# Get pRanges if not provided
	if pRanges is None:
		pRanges = createPRange(xRange, yRange, zRange)
		if verbose:
			print(f"Computed pRanges: {pRanges}")
	
	if len(pRanges) == 0:
		print("Warning: No valid pRanges for this range combination")
		return None
	
	# Collect all triangles from createtriangles
	index = []
	for pRange in pRanges:
		if verbose:
			print(f"\nProcessing pRange: {pRange}")
		createtriangles(xRange, yRange, zRange, pRange, index, debug=False)
	
	if len(index) == 0:
		print("Warning: No triangles found for this range combination")
		return None
	
	# Find the triangle with maximum ratio
	# Format: [idx, idy, idz, x, y, z, p, max_ratio, costIdx, -1111, ...]
	max_ratio = -100
	best_triangle = None
	best_idx = -1
	
	for i, triangle in enumerate(index):
		if len(triangle) >= 8:
			ratio_val = triangle[7]  # max_ratio is at index 7
			if ratio_val > max_ratio:
				max_ratio = ratio_val
				best_triangle = triangle
				best_idx = i
	
	if best_triangle is None:
		print("Warning: No valid triangle found")
		return None
	
	# Extract triangle information
	idx, idy, idz = best_triangle[0], best_triangle[1], best_triangle[2]
	x, y, z, p = best_triangle[3], best_triangle[4], best_triangle[5], best_triangle[6]
	costIdx = best_triangle[8]
	edge_type = EDGES[costIdx]
	
	# Get covariance values if available
	covxy, covxz, covyz = None, None, None
	if len(best_triangle) >= 22:
		covxy, covxz, covyz = best_triangle[19], best_triangle[20], best_triangle[21]
	
	result = {
		'x': x,
		'y': y,
		'z': z,
		'p': p,
		'max_ratio': max_ratio,
		'edge_type': edge_type,
		'edge_type_idx': costIdx,
		'range_indices': (idx, idy, idz),
		'covxy': covxy,
		'covxz': covxz,
		'covyz': covyz,
		'total_triangles': len(index)
	}
	
	# Print results
	if verbose:
		print("\n" + "="*80)
		print("RESULTS:")
		print("="*80)
		print(f"Maximum ratio: {max_ratio:.10f}")
		print(f"Best triangle:")
		print(f"  x = {x:.10f}")
		print(f"  y = {y:.10f}")
		print(f"  z = {z:.10f}")
		print(f"  p = {p:.10f}")
		print(f"  Edge type: {edge_type} (index {costIdx})")
		print(f"  Range indices: ({idx}, {idy}, {idz})")
		if covxy is not None:
			print(f"  Covariances: covxy={covxy:.10f}, covxz={covxz:.10f}, covyz={covyz:.10f}")
		print(f"\nTotal triangles generated: {len(index)}")
		print("="*80)
	
	return result

if __name__ == '__main__':
	start_time = time.time()
	f = [0, 0, 0, 0]

	print("ratio: ", target)

	# createtriangles(0.5, 0.5, 1, 0)
	# createtriangles(0.392, 0.392, 0.392, 0.216)
	# Example usage of compute_max_ratio_in_range function
	# Range = [0.48, 0.5, 0.48, 0.5, 0.48, 0.5, 0.25, 0.385]
	Range = [0.48, 0.5, 0.48, 0.5, 0.48, 0.5, 0.25, 0.385]
	Range = [0.9, 0.95, 0.95, 1.0, 0.95, 1.0, 0.0, 0.050000000000000044]
	Range = [0.44999999999999996, 0.5, 0.44999999999999996, 0.5, 0.95, 1.0, 0.0, 0.050000000000000044]
	Range = [0.48, 0.49, 0.48, 0.49, 0.48, 0.49, 0.495, 0.52]
	xRange, yRange, zRange, pRange = Range[0:2], Range[2:4], Range[4:6], Range[6:8]
	
	# Call the function to compute maximum ratio in the given range
	# check_ratio_for_range calls createtriangles from create_triangles.py
	# pRanges can be None (auto-compute) or a list of ranges
	pRanges = [pRange]  # Wrap single range in list, or set to None to auto-compute
	result = check_ratio_for_range(xRange, yRange, zRange, pRanges, verbose=True)

	# compute_triangle_ratio(0.4, 0.4, 0.4, 0.4, True)
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
