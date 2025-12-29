# Correlation Clustering LP

This project contains code to verify paper results for correlation clustering using cluster LP.

## Overview

The repository contains implementations to check two different LP formulations:

1. **Pure Cluster LP** (`pure_cluster_lp/`): Standard correlation clustering LP formulation
2. **Extended LP with 3-Strong-Adjacency** (`pure_cluster_lp_extended_3_SA/`): Enhanced LP formulation with additional constraints and p-value adjustments

## Installation

### Local Installation

For local execution, install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy scipy cvxpy
```

**Note**: If you have access to the [Mosek solver](https://www.mosek.com/), it is recommended for faster computation. See [docs/SETUP_ENVIRONMENT.md](docs/SETUP_ENVIRONMENT.md) for detailed installation instructions.

### Server/Cluster Installation

For server/cluster execution, see [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md) for complete instructions on environment setup and running jobs.

## Project Structure

```
SDP1437code/
├── pure_cluster_lp/                    # Pure cluster LP implementation
│   ├── create_triangles.py            # Generate triangle constraints
│   ├── sdp.py                         # Solve SDP and check approximation ratio
│   └── check_ratio.py                 # Check ratio for specific ranges
│
├── pure_cluster_lp_extended_3_SA/      # Extended 3-SA LP implementation
│   ├── create_triangles.py            # Generate triangles with p-value adjustments
│   ├── sdp.py                         # Solve SDP
│   ├── check_ratio.py                 # Check ratio utilities
│   └── debug_negative_ratio.py        # Debugging tools
│
├── pure_cluster_lp_server/             # Server scripts for pure cluster LP
│   ├── run_create_triangles_server.py
│   └── submit_slurm_*.sh
│
├── pure_cluster_lp_extended_3_SA_server/  # Server scripts for extended 3-SA
│   ├── run_create_triangles_server.py
│   └── submit_slurm_*.sh
│
├── docs/                                # Documentation
│   └── SERVER_GUIDE.md                 # Complete server setup and execution guide
│
├── utils/                              # Utility scripts
│   ├── test_solvers.py                 # Test CVXPY and MOSEK installation
│   └── check_ratio_mixed_rounding.py   # Mixed rounding ratio checker
│
├── requirements.txt                    # Python dependencies for local execution
└── requirements-server.txt             # Python dependencies for server/cluster
```

### Key Differences

**Pure Cluster LP** (`pure_cluster_lp/`):
- Standard correlation clustering LP formulation
- Does not adjust p-values based on `1 - (x+y+z)/2` constraint
- Uses original pl and pu values directly

**Extended 3-SA LP** (`pure_cluster_lp_extended_3_SA/`):
- Enhanced LP formulation with p-value adjustments
- Adjusts pl and pu to respect `p_min = 1 - (x+y+z)/2` constraint
- Recalculates ratios for boundary triangles

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

### For Extended 3-SA LP:

```bash
cd pure_cluster_lp_extended_3_SA
python create_triangles.py
python sdp.py triangles_merged.csv .
```

### Server Execution (SLURM):

See [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md) for complete instructions on environment setup and running jobs on a cluster.

## Parameters

- **target**: The approximation ratio you want to achieve
- **splitting point**: Splits the interval (for interval-based method)

## Verification

The `sdp.py` files check `-OPT_{SDP}`, which should be nearly 0 for valid solutions.

## Notes

- **Pure Cluster LP**: Uses range-based indexing for triangles, allowing the same (x,y,z) values in different ranges to have different IDs
- **Extended 3-SA LP**: Includes p-value adjustments to ensure `p >= 1 - (x+y+z)/2` for better ratio calculations
- Both implementations support parallel execution on SLURM clusters
- Results are saved in respective `output.txt` files
- Generated triangle files (CSV) are gitignored to keep the repository clean

## Requirements

- Python 3.x
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- CVXPY >= 1.2.0
- SCS solver (open-source, included with CVXPY)
- MOSEK (optional, but recommended for faster solving)

See `requirements.txt` for local installation or `requirements-server.txt` for server/cluster installation.

For detailed installation instructions, especially for server/cluster environments, see [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md).


