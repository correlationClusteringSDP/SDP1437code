#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT_NAME  # TODO: Replace with your SLURM account name

#SBATCH --job-name=sdp_solve_lp
#SBATCH --output=pure_cluster_lp_server/slurm_output/sdp_%j.out
#SBATCH --error=pure_cluster_lp_server/slurm_output/sdp_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com  # TODO: Replace with your email address
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32

# Load conda
module load anaconda3/2024.02
# Enable conda activate
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
# Activate your environment
conda activate /scratch/$USER/conda-envs/traffic_env

# Get the project root directory
# SLURM sets SLURM_SUBMIT_DIR to the directory where sbatch was called from
# This should be the project root directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback: try to get script directory
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Set MOSEK license path (if using MOSEK solver)
# Uncomment and adjust path if needed:
# export MOSEKLM_LICENSE_FILE=~/mosek.lic
# Or if license is in project directory:
# export MOSEKLM_LICENSE_FILE=$PROJECT_ROOT/mosek.lic

# Stay at project root directory
cd "$PROJECT_ROOT" || exit 1

# Note: SLURM will create output files automatically based on --output and --error directives
# No need to create directories manually - SLURM handles it

# Default input file (can be overridden by command line argument)
# Default assumes triangles_merged.csv is in pure_cluster_lp_server/
TRIANGLES_FILE=${1:-pure_cluster_lp_server/triangles_merged.csv}
OUTPUT_DIR=${2:-pure_cluster_lp}

# Convert relative paths to absolute if needed (relative to project root)
if [[ "$TRIANGLES_FILE" != /* ]]; then
    # If relative path, make it relative to project root
    TRIANGLES_FILE="$PROJECT_ROOT/$TRIANGLES_FILE"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    # If relative path, make it relative to project root
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi

echo "Working directory: $(pwd)"
echo "Triangles file: $TRIANGLES_FILE"
echo "Output directory: $OUTPUT_DIR"

# Check if triangles file exists
if [ ! -f "$TRIANGLES_FILE" ]; then
    echo "Error: Triangles file not found: $TRIANGLES_FILE"
    echo "Please provide a valid triangles CSV file."
    echo "Usage: sbatch submit_slurm_sdp.sh [triangles_file] [output_dir]"
    echo "Example: sbatch submit_slurm_sdp.sh pure_cluster_lp_server/triangles_merged.csv pure_cluster_lp"
    exit 1
fi

# Get file size for information
FILE_SIZE=$(du -h "$TRIANGLES_FILE" | cut -f1)
echo "Triangles file size: $FILE_SIZE"

# Count lines (triangles) for information
TRIANGLE_COUNT=$(wc -l < "$TRIANGLES_FILE")
echo "Number of triangles: $TRIANGLE_COUNT"

# Memory requirements note
echo ""
echo "Memory allocated: 60GB (estimated requirement for ~150K triangles)"
echo "If you have more triangles or larger numofEdgeRangeTypes, consider increasing --mem"

# Run SDP solver
# sdp.py is in pure_cluster_lp/ directory
# Redirect stderr to stdout so solver verbose output goes to .out file instead of .err
echo ""
echo "Starting SDP solver..."
python3 pure_cluster_lp/sdp.py "$TRIANGLES_FILE" "$OUTPUT_DIR" 2>&1

echo "SDP solver completed"

