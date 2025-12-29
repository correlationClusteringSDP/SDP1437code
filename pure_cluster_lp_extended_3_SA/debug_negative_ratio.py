#!/usr/bin/env python3
"""
Debug script to check why triangles have negative ratios
"""
import sys
sys.path.insert(0, '.')

from create_triangles import *


def test_ratio(x, y, z, p, xRange, yRange, zRange, verbose=True):
    """
    Test and print ratio for a specific triangle configuration.
    
    Parameters:
        x, y, z, p: Triangle parameters
        xRange, yRange, zRange: Ranges for the triangle
        verbose: If True, print detailed information
    """
    if verbose:
        print("="*80)
        print(f"Testing ratio for triangle: x={x}, y={y}, z={z}, p={p}")
        print("="*80)
        print(f"Ranges: xRange={xRange}, yRange={yRange}, zRange={zRange}")
        print()
    
    # Check constraints
    if verbose:
        print("Checking constraints:")
        check1_xy = check1(x, y, z)
        check1_xz = check1(z, x, y)
        check1_yz = check1(y, z, x)
        print(f"  check1(x, y, z) = {check1_xy} (x + y < z: {x + y < z})")
        print(f"  check1(z, x, y) = {check1_xz} (z + x < y: {z + x < y})")
        print(f"  check1(y, z, x) = {check1_yz} (y + z < x: {y + z < x})")
        
        check2_result = check2(x, y, z, p)
        print(f"  check2(x, y, z, p) = {check2_result}")
        print()
    
    # Check p_min constraint
    p_min = 1 - (x + y + z) / 2
    if verbose:
        print(f"p_min = 1 - (x+y+z)/2 = 1 - {x+y+z}/2 = {p_min}")
        print(f"p = {p} >= p_min = {p_min}: {p >= p_min}")
        print()
    
    # Calculate ratios for all edge types
    if verbose:
        print("Ratios for all edge types:")
    ratios = []
    max_ratio = -100
    best_edge = -1
    for i, edges in enumerate(EDGES):
        r = ratio(x, y, z, p, xRange, yRange, zRange, edges)
        ratios.append(r)
        if verbose:
            print(f"  Edge type {i} {edges}: ratio = {r:.10f}")
        if r > max_ratio:
            max_ratio = r
            best_edge = i
    
    if verbose:
        print(f"\nMaximum ratio: {max_ratio:.10f} (edge type {best_edge})")
        print()
    
    return {
        'x': x, 'y': y, 'z': z, 'p': p,
        'xRange': xRange, 'yRange': yRange, 'zRange': zRange,
        'p_min': p_min,
        'ratios': ratios,
        'max_ratio': max_ratio,
        'best_edge': best_edge,
        'all_ratios': dict(zip(range(len(EDGES)), ratios))
    }


def debug_createtriangles(xRange, yRange, zRange, pRanges=None, debug=True):
    """
    Debug createtriangles function to see what ratios are actually saved.
    
    Parameters:
        xRange, yRange, zRange: Ranges for the triangle
        pRanges: List of p ranges to test. If None, will compute using createPRange
        debug: If True, enable debug mode in createtriangles
    """
    print("="*80)
    print("Debugging createtriangles function")
    print("="*80)
    print(f"Ranges: xRange={xRange}, yRange={yRange}, zRange={zRange}")
    print()
    
    # Get pRanges if not provided
    if pRanges is None:
        pRanges = createPRange(xRange, yRange, zRange)
        print(f"Computed pRanges: {pRanges}")
        print()
    
    # Call createtriangles for each pRange
    index = []
    for pRange in pRanges:
        print(f"\nCalling createtriangles with pRange={pRange} (debug={debug}):")
        print("-" * 80)
        prev_count = len(index)
        createtriangles(xRange, yRange, zRange, pRange, index, debug=debug)
        new_count = len(index) - prev_count
        print("-" * 80)
        print(f"  Triangles added: {new_count} (total: {len(index)})")
    
    # Analyze saved triangles
    print(f"\nTotal triangles in index: {len(index)}")
    print("\nAll saved triangles:")
    for i, triangle in enumerate(index):
        # Format: [idx, idy, idz, x, y, z, p, max_ratio, costIdx, -1111, ...]
        if len(triangle) >= 8 and triangle[7] > 0:
            idx, idy, idz = triangle[0], triangle[1], triangle[2]
            x, y, z, p = triangle[3], triangle[4], triangle[5], triangle[6]
            max_ratio = triangle[7]
            costIdx = triangle[8]
            print(f"  Triangle {i}: x={x:.6f}, y={y:.6f}, z={z:.6f}, p={p:.6f}, "
                  f"max_ratio={max_ratio:.10f}, costIdx={costIdx}, "
                  f"range_idx=({idx}, {idy}, {idz})")
    
    return index


if __name__ == '__main__':
    # # Example 1: Test ratio for x=0.5, y=0.5, z=1, p=0
    # print("\n" + "="*80)
    # print("EXAMPLE 1: Testing ratio for x=0.5, y=0.5, z=1, p=0")
    # print("="*80)
    # result1 = test_ratio(0.5, 0.5, 1.0, 0.0, 
    #                     xRange=[0.4, 0.6], 
    #                     yRange=[0.4, 0.6], 
    #                     zRange=[0.8, 1.0],
    #                     verbose=True)
    
    # # Test with p = p_min
    # print("\n" + "-"*80)
    # print("Testing with p = p_min:")
    # print("-"*80)
    # result1_pmin = test_ratio(0.5, 0.5, 1.0, result1['p_min'],
    #                           xRange=[0.4, 0.6], 
    #                           yRange=[0.4, 0.6], 
    #                           zRange=[0.8, 1.0],
    #                           verbose=True)
    
    # # Example 2: Test ratio for x=0.4, y=0.4, z=0.4, p=0.0
    # print("\n" + "="*80)
    # print("EXAMPLE 2: Testing ratio for x=0.4, y=0.4, z=0.4, p=0.0")
    # print("="*80)
    # result2 = test_ratio(0.4, 0.4, 0.4, 0.0,
    #                     xRange=[0.4, 0.6],
    #                     yRange=[0.4, 0.6],
    #                     zRange=[0.4, 0.6],
    #                     verbose=True)
    
    # Example 3: Debug createtriangles for xRange=[0.4, 0.6], yRange=[0.4, 0.6], zRange=[0.4, 0.6]
    print("\n" + "="*80)
    print("EXAMPLE 3: Debugging createtriangles")
    print("="*80)
    triangles = debug_createtriangles(
        xRange=[0.4, 0.6],
        yRange=[0.4, 0.6],
        zRange=[0.8, 1],
        debug=False
    )
    
    # # Filter triangles with x=y=z=0.4
    # print("\n" + "="*80)
    # print("Triangles with x=y=z=0.4:")
    # print("="*80)
    # for i, triangle in enumerate(triangles):
    #     if len(triangle) >= 8:
    #         x, y, z = triangle[3], triangle[4], triangle[5]
    #         if abs(x - 0.4) < 1e-10 and abs(y - 0.4) < 1e-10 and abs(z - 0.4) < 1e-10:
    #             p = triangle[6]
    #             max_ratio = triangle[7]
    #             costIdx = triangle[8]
    #             print(f"  Triangle {i}: x={x}, y={y}, z={z}, p={p:.6f}, "
    #                   f"max_ratio={max_ratio:.10f}, costIdx={costIdx}")
