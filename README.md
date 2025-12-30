# Correlation Clustering LP

This project contains code to verify paper results for correlation clustering using cluster LP.

## Overview

The repository contains implementations to check two different LP formulations:

1. **Pure Cluster LP** (`pure_cluster_lp/`): Standard correlation clustering LP formulation
2. **Extended LP with 3-Sherali-Adams** (`pure_cluster_lp_extended_3_SA/`): Enhanced LP formulation with additional constraints and p-value adjustments

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

**Note**: If you have access to the [Mosek solver](https://www.mosek.com/), it is recommended for faster computation. See [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md) for detailed installation instructions.

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
├── pure_cluster_lp_extended_3_SA/      # Extended 3-Sherali-Adams LP implementation
│   ├── create_triangles.py            # Generate triangles with p-value adjustments
│   ├── sdp.py                         # Solve SDP
│   ├── check_ratio.py                 # Check ratio utilities
│   └── debug_negative_ratio.py        # Debugging tools
│
├── pure_cluster_lp_server/             # Server scripts for pure cluster LP
│   ├── run_create_triangles_server.py
│   └── submit_slurm_*.sh
│
├── pure_cluster_lp_extended_3_SA_server/  # Server scripts for extended 3-Sherali-Adams
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

**Extended 3-Sherali-Adams LP** (`pure_cluster_lp_extended_3_SA/`):
- Enhanced LP formulation with p-value adjustments
- Adjusts pl and pu to respect `p_min = 1 - (x+y+z)/2` constraint
- Recalculates ratios for boundary triangles
- Note: "3-SA" stands for "3-Sherali-Adams"

## Usage

### General Workflow

1. **Generate triangles**: Run the appropriate `create_triangles*.py` script
2. **Solve the LP**: Run the corresponding `sdp*.py` script

### For Pure Cluster LP:

```bash
cd pure_cluster_lp

# Option 1: Generate triangles_merged.csv locally
python create_triangles.py  # Generates triangles_merged.csv

# Option 2: Download pre-generated file from Google Drive
# See "Data Files" section below for download link

# Run SDP solver
python sdp.py  # Uses triangles_merged.csv by default
```

### For Extended 3-Sherali-Adams LP:

```bash
cd pure_cluster_lp_extended_3_SA

# Option 1: Generate triangles_merged.csv locally
python create_triangles.py  # Generates triangles_merged.csv

# Option 2: Download pre-generated file from Google Drive
# See "Data Files" section below for download link

# Run SDP solver
python sdp.py  # Uses triangles_merged.csv by default
```

### Server Execution (SLURM):

See [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md) for complete instructions on environment setup and running jobs on a cluster.

## Parameters

You can customize the following parameters in `create_triangles.py`:

### Key Parameters

- **`target`**: The approximation ratio you want to achieve
  - `pure_cluster_lp/create_triangles.py`: Default is `1.484`
  - `pure_cluster_lp_extended_3_SA/create_triangles.py`: Default is `1.437`
  - Located at the top of the file (line ~22)

- **`splittingPoints`**: List of values that define the ranges for triangle generation
  - Controls the granularity of the search space
  - More splitting points = finer granularity but more computation
  - Located at the top of the file (line ~38)
  - Example: `[0.0, 0.05, 0.1, 0.2, 0.3, ..., 1.0]`

- **`posThreshold1` and `posThreshold2`**: Threshold values for the rounding function
  - `pure_cluster_lp/create_triangles.py`: `posThreshold1 = 0.40`, `posThreshold2 = 0.57`
  - `pure_cluster_lp_extended_3_SA/create_triangles.py`: `posThreshold1 = 0.33`, `posThreshold2 = 1.1`
  - Located at the top of the file (line ~14-15)

- **`base`**: Base value for discretization (default: `20`)
  - Controls the resolution of the search

### How to Modify

1. Open the appropriate `create_triangles.py` file:
   - `pure_cluster_lp/create_triangles.py` for Pure Cluster LP
   - `pure_cluster_lp_extended_3_SA/create_triangles.py` for Extended 3-Sherali-Adams LP

2. Modify the parameters at the top of the file (in the "Algorithm parameter" section)

3. Regenerate triangles by running `create_triangles.py`

**Note**: Changing these parameters will affect the generated triangles and may require regenerating `triangles_merged.csv`.

## Verification

The `sdp.py` files check `-OPT_{SDP}`, which should be nearly 0 for valid solutions.

## Data Files

The generated `triangles_merged.csv` files are large (~35MB each) and are not included in this repository. You can:

1. **Generate them locally**: Run `create_triangles.py` in the respective directories
2. **Download pre-generated files**: Available on [Google Drive](https://drive.google.com/drive/folders/1iW3r3z1ntlhA_-rD9I2jyDwq6ddenrBp?usp=drive_link)
   - `pure_cluster_lp/triangles_merged.csv`
   - `pure_cluster_lp_extended_3_SA/triangles_merged.csv`

After downloading, place the files in their respective directories before running `sdp.py`.

## Notes

- **Pure Cluster LP**: Uses range-based indexing for triangles, allowing the same (x,y,z) values in different ranges to have different IDs
- **Extended 3-Sherali-Adams LP**: Includes p-value adjustments to ensure `p >= 1 - (x+y+z)/2` for better ratio calculations
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


