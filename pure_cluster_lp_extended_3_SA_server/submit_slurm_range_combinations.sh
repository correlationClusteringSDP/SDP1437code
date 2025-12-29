#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT_NAME  # TODO: Replace with your SLURM account name

#SBATCH --job-name=triangles_range
#SBATCH --output=pure_cluster_lp_extended_3_SA_server/slurm_output/triangles_range_%A_%a.out
#SBATCH --error=pure_cluster_lp_extended_3_SA_server/slurm_output/triangles_range_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com  # TODO: Replace with your email address
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

#SBATCH --array=0-999%200

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

# Stay at project root directory
cd "$PROJECT_ROOT" || exit 1

# Note: SLURM will create output files automatically based on --output and --error directives
# Output directories will be created automatically when files are written

# Debug: Show current directory and check for files
echo "Current working directory: $(pwd)"
echo "Project root: $PROJECT_ROOT"
echo "Checking for required files..."
echo ""
echo "Listing pure_cluster_lp_extended_3_SA_server directory:"
if [ -d "pure_cluster_lp_extended_3_SA_server" ]; then
    ls -la pure_cluster_lp_extended_3_SA_server/ | head -20
else
    echo "  Directory 'pure_cluster_lp_extended_3_SA_server' not found!"
fi
echo ""

# Get current task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Total number of servers (calculated from array range, or set manually)
# If using --array=0-999, then total servers is 1000
TOTAL_SERVERS=1000

# Check if required files exist (in pure_cluster_lp_extended_3_SA_server directory)
REQUIRED_FILES=("pure_cluster_lp_extended_3_SA_server/range_combinations.txt" "pure_cluster_lp_extended_3_SA_server/splitting_points.txt" "pure_cluster_lp_extended_3_SA_server/edges_index.txt")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
        echo "  Missing: $file"
        # Also check if file exists with absolute path
        abs_file="$PROJECT_ROOT/$file"
        if [ -f "$abs_file" ]; then
            echo "    But found at: $abs_file"
        fi
        # Check if file exists in current directory structure
        if [ -f "$(basename "$file")" ]; then
            echo "    But found as: $(basename "$file") in $(pwd)"
        fi
    else
        echo "  Found: $file"
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "Error: Required files not found:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "To generate these files, run from project root:"
    echo "  cd pure_cluster_lp_extended_3_SA_server"
    echo "  python3 run_create_triangles_server.py --generate"
    echo "  cd .."
    echo ""
    echo "Or from project root:"
    echo "  cd pure_cluster_lp_extended_3_SA_server && python3 run_create_triangles_server.py --generate && cd .."
    echo ""
    echo "Current working directory: $(pwd)"
    echo "Looking for files relative to: $PROJECT_ROOT"
    exit 1
fi

echo "Task ID: $TASK_ID"
echo "Total servers: $TOTAL_SERVERS"
echo "Required files found:"
for file in "${REQUIRED_FILES[@]}"; do
    echo "  - $file"
done

# Run Python script (server wrapper that calls create_triangles from parent directory)
python3 pure_cluster_lp_extended_3_SA_server/run_create_triangles_server.py $TASK_ID $TOTAL_SERVERS pure_cluster_lp_extended_3_SA_server/range_combinations.txt pure_cluster_lp_extended_3_SA_server/output_triangles

echo "Task $TASK_ID completed"

