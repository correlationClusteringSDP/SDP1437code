# Server Setup and Execution Guide

Complete guide for setting up the environment and running the project on a server/cluster.

## Part 1: Environment Setup

### Prerequisites

- Access to the server/cluster
- Conda environment `traffic_env` already created at `/scratch/$USER/conda-envs/traffic_env`
- MOSEK license file (`mosek.lic`) - optional but recommended

### Step 1: Activate Conda Environment

```bash
# Load conda module
module load anaconda3/2024.02

# Enable conda activate
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Activate your environment
conda activate /scratch/$USER/conda-envs/traffic_env
```

### Step 2: Install CVXPY and Dependencies

```bash
# Install numpy and scipy (if not already installed)
conda install -y numpy scipy

# Install CVXPY and SCS solver
conda install -y -c conda-forge cvxpy scs

# Or using pip:
# pip install cvxpy scs
```

**Verify installation:**
```bash
python3 -c "import cvxpy as cp; print('CVXPY version:', cp.__version__)"
```

### Step 3: Install MOSEK (Optional but Recommended)

```bash
# Install MOSEK
conda install -y -c mosek mosek

# Or using pip:
# pip install mosek
```

### Step 4: Set Up MOSEK License

Place your `mosek.lic` file in one of these locations:

**Option 1: Home directory (Recommended)**
```bash
cp /path/to/your/mosek.lic ~/mosek.lic
chmod 644 ~/mosek.lic
```

**Option 2: Environment variable**
```bash
export MOSEKLM_LICENSE_FILE=/path/to/your/mosek.lic
# Add to ~/.bashrc for permanent setup
```

**Option 3: Default MOSEK location**
```bash
mkdir -p ~/.mosek
cp /path/to/your/mosek.lic ~/.mosek/mosek.lic
chmod 644 ~/.mosek/mosek.lic
```

### Step 5: Test Installation

```bash
# Navigate to project root
cd /path/to/SDP1437code

# Run test script
python3 utils/test_solvers.py
```

This will verify CVXPY, SCS, and MOSEK (if installed) are working correctly.

## Part 2: Running Jobs on Server

### Step 0: Configure SLURM Scripts

Before submitting jobs, you need to configure the SLURM scripts with your account information:

1. **Edit SLURM scripts** to set your account name and email:
   ```bash
   # Edit the submit scripts
   nano pure_cluster_lp_extended_3_SA_server/submit_slurm_range_combinations.sh
   nano pure_cluster_lp_extended_3_SA_server/submit_slurm_sdp.sh
   ```

2. **Replace placeholders** in the scripts:
   - `YOUR_ACCOUNT_NAME` → Your SLURM account name (e.g., `pr_368_general`)
   - `YOUR_EMAIL@example.com` → Your email address for job notifications

   Look for these lines:
   ```bash
   #SBATCH --account=YOUR_ACCOUNT_NAME  # TODO: Replace with your SLURM account name
   #SBATCH --mail-user=YOUR_EMAIL@example.com  # TODO: Replace with your email address
   ```

3. **Optional**: If you don't want email notifications, you can comment out or remove the mail lines:
   ```bash
   # #SBATCH --mail-type=ALL
   # #SBATCH --mail-user=YOUR_EMAIL@example.com
   ```

### Step 1: Connect to Server

```bash
# SSH to your server
ssh your_username@server_address

# Navigate to the project root directory
cd /path/to/SDP1437code
```

**Important**: All scripts run from the project root directory (`SDP1437code`), not from subdirectories.

### Step 2: Generate Range Combinations (First Time Only)

Before submitting jobs, generate the range combinations and configuration files:

```bash
# Run from project root directory
python3 pure_cluster_lp_extended_3_SA_server/run_create_triangles_server.py --generate
```

This creates in `pure_cluster_lp_extended_3_SA_server/`:
- `range_combinations.txt` - All valid (i, j, k) combinations
- `splitting_points.txt` - Splitting points used
- `edges_index.txt` - Edge index mappings

**Note**: 
- This step only needs to be done once. All servers will use these files.
- You can run this command from the project root directory - no need to `cd` into subdirectories.

### Step 3: Check Generated Files

```bash
ls -lh pure_cluster_lp_extended_3_SA_server/range_combinations.txt \
      pure_cluster_lp_extended_3_SA_server/splitting_points.txt \
      pure_cluster_lp_extended_3_SA_server/edges_index.txt

# Check number of combinations
wc -l pure_cluster_lp_extended_3_SA_server/range_combinations.txt
```

### Step 4: Modify SLURM Script (if needed)

Edit `submit_slurm_range_combinations.sh` to adjust:
- **Array range**: `--array=0-999%200` (1000 servers, max 200 concurrent)
- **Total servers**: `TOTAL_SERVERS=1000` (must match array range)
- **Resources**: `--mem=32G`, `--time=24:00:00`, `--cpus-per-task=16`

### Step 5: Submit SLURM Job

From the project root directory:

```bash
sbatch pure_cluster_lp_extended_3_SA_server/submit_slurm_range_combinations.sh
```

You should see: `Submitted batch job 12345`

### Step 6: Monitor Jobs

```bash
# View all your jobs
squeue -u $USER

# View job output
cat pure_cluster_lp_extended_3_SA_server/slurm_output/triangles_range_<JOB_ID>_<TASK_ID>.out

# Check progress
ls pure_cluster_lp_extended_3_SA_server/output_triangles/triangles_server_*.csv | wc -l
```

### Step 7: Merge Results

After all servers complete:

```bash
python3 pure_cluster_lp_extended_3_SA_server/run_create_triangles_server.py --merge \
    pure_cluster_lp_extended_3_SA_server/output_triangles \
    pure_cluster_lp_extended_3_SA_server/triangles_merged.csv
```

### Step 8: Run SDP Solver

The SDP solver requires significant memory (default: 60GB):

```bash
# Submit SDP job
sbatch pure_cluster_lp_extended_3_SA_server/submit_slurm_sdp.sh \
    pure_cluster_lp_extended_3_SA_server/triangles_merged.csv \
    pure_cluster_lp_extended_3_SA

# Check results after completion
cat pure_cluster_lp_extended_3_SA/output.txt
```

**Memory requirements:**
- For ~150K triangles: ~60GB recommended
- Memory scales quadratically with number of edge range types
- Adjust `--mem` in `submit_slurm_sdp.sh` if needed

## Troubleshooting

### Installation Issues

**CVXPY installation fails:**
```bash
# Try pip instead
pip install cvxpy scs
```

**MOSEK license not found:**
```bash
# Check license location
ls -la ~/mosek.lic
echo $MOSEKLM_LICENSE_FILE

# Test license
python3 -c "import mosek; env = mosek.Env(); print(env.getlicensepath())"
```

### Job Execution Issues

**Jobs fail to start:**
- Check required files exist: `range_combinations.txt`, `splitting_points.txt`, `edges_index.txt`
- Verify SLURM account name in script
- Test conda environment: `conda activate /scratch/$USER/conda-envs/traffic_env`

**Jobs run out of memory:**
- Increase `--mem` in SLURM script
- Filter triangles to reduce problem size
- Use MOSEK solver (often more memory efficient)

**Some tasks fail:**
- Check error logs: `cat pure_cluster_lp_extended_3_SA_server/slurm_output/triangles_range_<JOB_ID>_<TASK_ID>.err`
- Rerun failed tasks manually, then merge again

## Quick Reference

**Complete workflow:**
```bash
# 1. Setup environment (one time)
module load anaconda3/2024.02
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /scratch/$USER/conda-envs/traffic_env
conda install -y -c conda-forge cvxpy scs numpy scipy
conda install -y -c mosek mosek  # Optional
python3 utils/test_solvers.py  # Verify installation

# 2. Navigate to project root (if not already there)
cd /path/to/SDP1437code

# 3. Generate combinations (one time, from project root)
python3 pure_cluster_lp_extended_3_SA_server/run_create_triangles_server.py --generate

# 4. Submit triangle generation jobs (from project root)
sbatch pure_cluster_lp_extended_3_SA_server/submit_slurm_range_combinations.sh

# 5. Wait for completion, then merge (from project root)
python3 pure_cluster_lp_extended_3_SA_server/run_create_triangles_server.py --merge \
    pure_cluster_lp_extended_3_SA_server/output_triangles \
    pure_cluster_lp_extended_3_SA_server/triangles_merged.csv

# 6. Solve SDP (from project root)
sbatch pure_cluster_lp_extended_3_SA_server/submit_slurm_sdp.sh \
    pure_cluster_lp_extended_3_SA_server/triangles_merged.csv \
    pure_cluster_lp_extended_3_SA
```

## Additional Resources

- CVXPY documentation: https://www.cvxpy.org/
- MOSEK documentation: https://docs.mosek.com/
- MOSEK license setup: https://docs.mosek.com/9.4/install/installation.html#setting-up-a-license

