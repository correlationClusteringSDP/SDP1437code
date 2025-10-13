# Correlation Clustering LP

This project contains code to verify paper results for correlation clustering using cluster LP.

## Overview

The repository contains implementations to check two different LP formulations:

1. **Pure Cluster LP** (`pure_cluster_lp/`): Standard correlation clustering LP formulation
2. **Stronger LP with 3-Strong-Adjacency** (`cluster_lp_3_SA/`): Enhanced LP formulation with additional constraints

## Installation

Install the required Python package:

```bash
pip install cvxpy
```

**Note**: If you have access to the [Mosek solver](https://www.mosek.com/), it is recommended for faster computation.

## Project Structure

### pure_cluster_lp/
Contains files for the pure cluster LP formulation:
- `create_triangles.py`: Generates triangle constraints
- `sdp.py`: Solves the SDP and checks the approximation ratio
- `triangles.csv`: Generated triangle data
- `output.txt`: Results

### cluster_lp_3_SA/
Contains files for the stronger LP with 3-Strong-Adjacency constraints. This folder includes **two different implementations**:

#### Implementation 1: Interval-based (Multiple Occurrences)
- `create_triangles_interval.py`: Generates triangles using interval identification
- `sdp_interval.py`: Solves the LP using interval method
- **Note**: Each triangle may appear up to 8 times in this implementation

#### Implementation 2: Unique Triangles (Recommended)
- `create_triangles_with_3SA.py`: Generates triangles where each appears exactly once
- `sdp_with_3SA.py`: Solves the LP with unique triangle representation
- `compute_triangle_ratio.py`: Additional ratio computation utilities
- `check_ratio_test.py`: Test ratio verification
- **Note**: Each triangle appears only once, more efficient

## Usage

### General Workflow

1. **Generate triangles**: Run the appropriate `create_triangles*.py` script
2. **Solve the LP**: Run the corresponding `sdp*.py` script

### For Pure Cluster LP:

```bash
cd pure_cluster_lp
python create_triangles.py
python sdp.py
```

### For Stronger LP (Interval-based):

```bash
cd cluster_lp_3_SA
python create_triangles_interval.py
python sdp_interval.py
```

### For Stronger LP (Unique triangles):

```bash
cd cluster_lp_3_SA
python create_triangles_with_3SA.py
python sdp_with_3SA.py
```

## Parameters

- **target**: The approximation ratio you want to achieve
- **splitting point**: Splits the interval (for interval-based method)

## Verification

The `sdp.py` files check `-OPT_{SDP}`, which should be nearly 0 for valid solutions.

## Notes

- Each implementation uses different methods to identify and count triangles
- The interval-based method may have redundancy (triangles appearing up to 8 times)
- The unique triangle method is generally more efficient, at the cost of the approximate ratio
- Results are saved in respective `output.txt` files


