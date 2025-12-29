#!/usr/bin/env python3
"""
Check ratio for a specific range by calling createtriangles
Outputs the maximum ratio and corresponding triangle
"""
import sys
import os
import time

# Import from create_triangles module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_triangles import (
    createtriangles, createPRange, EDGES, 
    check1, ratio, splittingPoints, filtersplittlingpoints
)

def check_ratio_for_range(xRange, yRange, zRange, pRanges=None, verbose=True):
    """
    Call createtriangles for a range and find the maximum ratio triangle.
    
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
    
    # Example usage: check ratio for a specific range
    # You can modify these ranges as needed
    
    # Example 1: Specific range
    xRange = [0.4, 0.405]
    yRange = [0.4, 0.405]
    zRange = [0.4, 0.405]
    pRanges = [[0.231,0.272]]  # Optional: specify pRange, or set to None to auto-compute
    
    # Example 2: Single point (small range)
    # xRange = [0.42, 0.42]
    # yRange = [0.42, 0.42]
    # zRange = [0.42, 0.42]
    # pRanges = None
    
    # Example 3: Larger range
    # xRange = [0.4, 0.5]
    # yRange = [0.4, 0.5]
    # zRange = [0.4, 0.5]
    # pRanges = None
    
    print("Checking ratio for range...")
    result = check_ratio_for_range(xRange, yRange, zRange, pRanges, verbose=True)
    
    if result:
        print(f"\nSummary:")
        print(f"  Best triangle: x={result['x']:.6f}, y={result['y']:.6f}, z={result['z']:.6f}, p={result['p']:.6f}")
        print(f"  Maximum ratio: {result['max_ratio']:.10f}")
        print(f"  Edge type: {result['edge_type']}")
    
    pRanges = [[0.559,0.6]]

    print("Checking ratio for range...")
    result = check_ratio_for_range(xRange, yRange, zRange, pRanges, verbose=True)
    
    if result:
        print(f"\nSummary:")
        print(f"  Best triangle: x={result['x']:.6f}, y={result['y']:.6f}, z={result['z']:.6f}, p={result['p']:.6f}")
        print(f"  Maximum ratio: {result['max_ratio']:.10f}")
        print(f"  Edge type: {result['edge_type']}")


    print(f"\n--- {time.time() - start_time:.2f} seconds ---")
