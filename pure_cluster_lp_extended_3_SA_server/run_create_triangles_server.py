#!/usr/bin/env python3
"""
Server wrapper script to call create_triangles.py from the parent directory
Handles server mode: reading range combinations and processing assigned tasks
Also generates range combinations if they don't exist
"""
import sys
import os
import csv
import time
import glob
from collections import defaultdict

# Add parent directory to path to import create_triangles
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pure_cluster_lp_extended_3_SA.create_triangles import (
    splittingPoints, filtersplittlingpoints,
    filter_index, edgesIndex, createPRange, createtriangles,
    roundIndex, getEdgeIndex, getEdgeRangeIndex, upper_base_from_arraydict
)

def initialize_edgesIndex(splittingPoints_list):
    """
    Initialize edgesIndex by pre-creating indices in the same order as the local version.
    The local version creates indices during construct_triangles() by calling getEdgeIndex.
    Note: Since we now use rangeBasedEdgesIndex for triangle indexing, edgesIndex is mainly
    kept for backward compatibility.
    """
    import pure_cluster_lp_extended_3_SA.create_triangles as ct_module
    
    # Initialize edgesIndex in the same order as construct_triangles() would create them
    # This matches the order in which getEdgeIndex is called
    n = len(splittingPoints_list) - 1
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                xRange = [splittingPoints_list[i], splittingPoints_list[i + 1]]
                yRange = [splittingPoints_list[j], splittingPoints_list[j + 1]]
                zRange = [splittingPoints_list[k], splittingPoints_list[k + 1]]
                if xRange[1] + yRange[1] <= zRange[0]:
                    break
                zRange[1] = min(xRange[1] + yRange[1], zRange[1])
                xRange[0] = max(xRange[0], zRange[0] - yRange[1])
                yRange[0] = max(yRange[0], zRange[0] - xRange[1])
                
                # Call getEdgeIndex for each boundary value
                for x in xRange:
                    getEdgeIndex(x)
                for y in yRange:
                    getEdgeIndex(y)
                for z in zRange:
                    getEdgeIndex(z)
    
    # Update the module's edgesIndex
    ct_module.edgesIndex = edgesIndex
    print(f"Initialized edgesIndex with {len(edgesIndex)} entries")

def save_edgesIndex(edgesIndex_dict, edges_index_file='edges_index.txt'):
    """
    Save edgesIndex to file for consistency across servers
    Format: roundIndex_value:edge_index
    """
    with open(edges_index_file, 'w') as f:
        # Sort by roundIndex value for consistency
        sorted_items = sorted(edgesIndex_dict.items())
        for round_idx, edge_idx in sorted_items:
            f.write(f"{round_idx}:{edge_idx}\n")
    print(f"edgesIndex saved to: {edges_index_file}")

def load_edgesIndex(edges_index_file='edges_index.txt'):
    """
    Load edgesIndex from file
    Returns the edgesIndex dictionary
    """
    edgesIndex_dict = {}
    if os.path.exists(edges_index_file):
        with open(edges_index_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    round_idx_str, edge_idx_str = line.split(':', 1)
                    round_idx = int(round_idx_str)
                    edge_idx = int(edge_idx_str)
                    edgesIndex_dict[round_idx] = edge_idx
        print(f"Loaded edgesIndex with {len(edgesIndex_dict)} entries from {edges_index_file}")
    else:
        print(f"Warning: {edges_index_file} not found")
    return edgesIndex_dict

def generate_range_combinations(splittingPoints_list, range_combinations_file=None, 
                                 splitting_points_file=None,
                                 edges_index_file=None):
    """
    Generate all possible (i, j, k) range combinations and save to file
    Also saves splitting points and edgesIndex to ensure consistency
    
    Files are saved in the server directory (pure_cluster_lp_extended_3_SA_server)
    A configuration summary is created in the project root directory
    """
    import pure_cluster_lp_extended_3_SA.create_triangles as ct_module
    
    # Determine project root and server directory
    # This script is in pure_cluster_lp_extended_3_SA_server/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = script_dir  # pure_cluster_lp_extended_3_SA_server
    project_root = os.path.dirname(script_dir)  # Project root (SDP1437code)
    
    # Set default file paths in server directory
    if range_combinations_file is None:
        range_combinations_file = os.path.join(server_dir, 'range_combinations.txt')
    if splitting_points_file is None:
        splitting_points_file = os.path.join(server_dir, 'splitting_points.txt')
    if edges_index_file is None:
        edges_index_file = os.path.join(server_dir, 'edges_index.txt')
    
    # Configuration file in project root
    config_file = os.path.join(project_root, 'server_config.txt')
    
    n = len(splittingPoints_list) - 1
    
    # Initialize edgesIndex before generating combinations (this uses the markers)
    initialize_edgesIndex(splittingPoints_list)
    
    # Generate all (i, j, k) combinations
    combinations = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                xRange = [splittingPoints_list[i], splittingPoints_list[i + 1]]
                yRange = [splittingPoints_list[j], splittingPoints_list[j + 1]]
                zRange = [splittingPoints_list[k], splittingPoints_list[k + 1]]
                # Check if valid (triangle constraint)
                if xRange[1] + yRange[1] <= zRange[0]:
                    break
                # This is a valid combination
                combinations.append((i, j, k))
    
    print(f"Total splitting points: {len(splittingPoints_list)}")
    print(f"Total range combinations: {len(combinations)}")
    
    # Ensure server directory exists
    os.makedirs(server_dir, exist_ok=True)
    
    # Save splitting points to file (in server directory)
    with open(splitting_points_file, 'w') as f:
        for point in splittingPoints_list:
            f.write(f"{point}\n")
    
    print(f"Splitting points saved to: {splitting_points_file}")
    
    # Save edgesIndex to file (in server directory)
    save_edgesIndex(edgesIndex, edges_index_file)
    
    # Save combination list to file (in server directory)
    with open(range_combinations_file, 'w') as f:
        for i, j, k in combinations:
            f.write(f"{i} {j} {k}\n")
    
    print(f"Range combinations list saved to: {range_combinations_file}")
    
    # Create configuration summary file in project root
    with open(config_file, 'w') as f:
        f.write("Server Configuration Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total splitting points: {len(splittingPoints_list)}\n")
        f.write(f"Total range combinations: {len(combinations)}\n")
        f.write(f"edgesIndex size: {len(edgesIndex)}\n\n")
        f.write("File Locations (relative to project root):\n")
        f.write(f"  - range_combinations.txt: pure_cluster_lp_extended_3_SA_server/range_combinations.txt\n")
        f.write(f"  - splitting_points.txt: pure_cluster_lp_extended_3_SA_server/splitting_points.txt\n")
        f.write(f"  - edges_index.txt: pure_cluster_lp_extended_3_SA_server/edges_index.txt\n\n")
        f.write("Splitting Points (first 10 and last 10):\n")
        if len(splittingPoints_list) <= 20:
            for point in splittingPoints_list:
                f.write(f"  {point}\n")
        else:
            for point in splittingPoints_list[:10]:
                f.write(f"  {point}\n")
            f.write("  ...\n")
            for point in splittingPoints_list[-10:]:
                f.write(f"  {point}\n")
    
    print(f"Configuration summary saved to: {config_file}")
    
    return combinations

def construct_triangles_for_ranges(range_combinations):
    """
    Construct triangles for specified range combinations list
    range_combinations: list of [(i, j, k), ...] tuples
    """
    # Use the imported splittingPoints from the module
    import pure_cluster_lp_extended_3_SA.create_triangles as ct_module
    splittingPoints_list = ct_module.splittingPoints
    
    # Validate splittingPoints_list
    if len(splittingPoints_list) < 2:
        raise ValueError(f"Invalid splittingPoints_list: need at least 2 points, got {len(splittingPoints_list)}")
    
    n = len(splittingPoints_list) - 1
    
    # pa is already initialized at module level in create_triangles.py
    # No need to create it here
    
    index = []
    total = len(range_combinations)
    skipped_invalid = 0
    
    for idx, (i, j, k) in enumerate(range_combinations):
        if idx % max(total // 10, 1) == 0:
            print(f"Progress: {100.0 * idx / total:.1f}% ({idx}/{total})")
        
        # Validate indices
        if i < 0 or i >= n or j < 0 or j >= n or k < 0 or k >= n:
            skipped_invalid += 1
            print(f"Warning: Invalid indices (i={i}, j={j}, k={k}), skipping. Valid range: 0-{n-1}")
            continue
        
        xRange = [splittingPoints_list[i], splittingPoints_list[i + 1]]
        yRange = [splittingPoints_list[j], splittingPoints_list[j + 1]]
        zRange = [splittingPoints_list[k], splittingPoints_list[k + 1]]
        
        # Apply triangle constraints
        if xRange[1] + yRange[1] <= zRange[0]:
            continue
        zRange[1] = min(xRange[1] + yRange[1], zRange[1])
        xRange[0] = max(xRange[0], zRange[0] - yRange[1])
        yRange[0] = max(yRange[0], zRange[0] - xRange[1])
        
        pRanges = createPRange(xRange, yRange, zRange)
        if len(pRanges) == 0:
            continue
        for pRange in pRanges:
            createtriangles(xRange, yRange, zRange, pRange, index)
    
    if skipped_invalid > 0:
        print(f"Warning: Skipped {skipped_invalid} invalid combinations")
    
    return index

def merge_triangles(output_dir="output_triangles", output_file="triangles_merged.csv"):
    """
    Merge all triangles_server_*.csv files from parallel server processing.
    Reads all triangles into a list and uses filter_index to remove duplicates.
    """
    # Find all result files (support different filename patterns)
    pattern1 = os.path.join(output_dir, "triangles_server_*.csv")
    pattern2 = os.path.join(output_dir, "triangles_i*_j*.csv")
    files1 = glob.glob(pattern1)
    files2 = glob.glob(pattern2)
    files = files1 + files2
    
    if not files:
        print(f"No result files found in {output_dir}")
        print(f"Looking for patterns: {pattern1} or {pattern2}")
        return
    
    print(f"Found {len(files)} result files")
    print("Reading files... (this may take a while)")
    
    # Collect all triangles into a list (same format as filter_index expects)
    index = []
    total_count = 0
    files_read = 0
    files_with_errors = 0
    
    for file_path in files:
        try:
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 7:
                        continue
                    total_count += 1
                    
                    # Convert CSV row (strings) to list of numbers/values
                    # New format: [idx, idy, idz, x, y, z, p, max_ratio, costIdx, -1111, ...]
                    try:
                        # Convert all values to appropriate types
                        converted_row = []
                        for i, val in enumerate(row):
                            if i < 3:
                                # First 3 are range indices (integers)
                                converted_row.append(int(float(val)))
                            else:
                                # Rest are floats (or integers like -1111, -2222)
                                converted_row.append(float(val))
                        index.append(converted_row)
                    except (ValueError, IndexError) as e:
                        # Skip invalid rows
                        print(f"Warning: Skipping invalid row in {file_path}: {e}")
                        continue
            files_read += 1
        except Exception as e:
            files_with_errors += 1
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Files processed: {files_read} successful, {files_with_errors} with errors")
    print(f"Total triangles read: {total_count}")
    
    # Use filter_index to remove duplicates (same logic as create_triangles.py)
    print("Filtering duplicates using filter_index...")
    filtered_triangles = filter_index(index)
    
    print(f"Final triangles after filtering: {len(filtered_triangles)}")
    
    # Write merged file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_triangles)
    
    print(f"Merged results written to: {output_file}")
    
    # Print edgesIndex information (if available)
    try:
        import pure_cluster_lp_extended_3_SA.create_triangles as ct_module
        print(f"edgesIndex: {ct_module.edgesIndex}")
    except:
        print("Note: edgesIndex information not available")

if __name__ == '__main__':
    start_time = time.time()
    
    # Check for merge mode
    if len(sys.argv) >= 2 and sys.argv[1] in ['--merge', '-m', 'merge']:
        # Merge mode: merge all result files
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_triangles"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "triangles_merged.csv"
        merge_triangles(output_dir, output_file)
        sys.exit(0)
    
    # Check if we should just generate combinations (no server arguments)
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--generate', '-g', 'generate']):
        # Generate mode: just create range combinations
        print("Generating range combinations...")
        print("Files will be saved in: pure_cluster_lp_extended_3_SA_server/")
        print("Configuration summary will be saved in project root: server_config.txt")
        splittingPoints_list = list(filter(filtersplittlingpoints, list(set(splittingPoints))))
        splittingPoints_list.sort()
        generate_range_combinations(splittingPoints_list)
        print("Done!")
        sys.exit(0)
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Generate range combinations:")
        print("    python3 run_create_triangles_server.py [--generate|-g]")
        print("  Server mode:")
        print("    python3 run_create_triangles_server.py <server_id> <total_servers> [range_combinations_file] [output_dir]")
        print("  Merge results:")
        print("    python3 run_create_triangles_server.py [--merge|-m] [output_dir] [output_file]")
        print("  For standalone mode, run create_triangles.py directly from pure_cluster_lp_extended_3_SA directory")
        sys.exit(1)
    
    # Server mode
    server_id = int(sys.argv[1])
    total_servers = int(sys.argv[2])
    
    # Determine project root and server directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = script_dir  # pure_cluster_lp_extended_3_SA_server
    project_root = os.path.dirname(script_dir)  # Project root (SDP1437code)
    
    # Default file paths in server directory
    default_range_combinations_file = os.path.join(server_dir, 'range_combinations.txt')
    default_splitting_points_file = os.path.join(server_dir, 'splitting_points.txt')
    default_edges_index_file = os.path.join(server_dir, 'edges_index.txt')
    
    range_combinations_file = sys.argv[3] if len(sys.argv) > 3 else default_range_combinations_file
    output_dir = sys.argv[4] if len(sys.argv) > 4 else 'output_triangles'
    
    # If relative path provided, assume it's relative to server directory
    if not os.path.isabs(range_combinations_file):
        if not os.path.exists(range_combinations_file):
            range_combinations_file = os.path.join(server_dir, range_combinations_file)
    
    print(f"Server mode: Server ID={server_id}, Total servers={total_servers}")
    
    # Load splitting points from file if exists (to ensure consistency with generated combinations)
    splitting_points_file = default_splitting_points_file
    if os.path.exists(splitting_points_file):
        print(f"Reading splitting points from: {splitting_points_file}")
        splittingPoints_list = []
        with open(splitting_points_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    splittingPoints_list.append(float(line))
        splittingPoints_list.sort()
        print(f"Loaded {len(splittingPoints_list)} splitting points from file")
    else:
        # If file doesn't exist, process splittingPoints (same as in create_triangles.py)
        print(f"Warning: {splitting_points_file} not found, using computed splitting points")
        splittingPoints_list = list(filter(filtersplittlingpoints, list(set(splittingPoints))))
        splittingPoints_list.sort()
    
    # Load splitting point markers from file if exists
    import pure_cluster_lp_extended_3_SA.create_triangles as ct_module
    
    # Load edgesIndex from file if exists (to ensure consistency)
    edges_index_file = default_edges_index_file
    if os.path.exists(edges_index_file):
        print(f"Reading edgesIndex from: {edges_index_file}")
        loaded_edgesIndex = load_edgesIndex(edges_index_file)
        # Update both local and module's edgesIndex
        edgesIndex.clear()
        edgesIndex.update(loaded_edgesIndex)
        ct_module.edgesIndex = edgesIndex
    else:
        print(f"Warning: {edges_index_file} not found, will initialize edgesIndex")
        # Initialize edgesIndex if file doesn't exist (using loaded markers)
        initialize_edgesIndex(splittingPoints_list)
    
    # Update the global splittingPoints (this is a bit hacky, but needed for the imported functions)
    ct_module.splittingPoints = splittingPoints_list
    
    # Validate splittingPoints_list length (should match expected range)
    if len(splittingPoints_list) < 2:
        print(f"Error: Invalid splittingPoints_list: need at least 2 points, got {len(splittingPoints_list)}")
        sys.exit(1)
    
    print(f"Splitting points count: {len(splittingPoints_list)}")
    print(f"edgesIndex size: {len(edgesIndex)}")
    
    # Validate that edgesIndex is not empty
    if len(edgesIndex) == 0:
        print("Warning: edgesIndex is empty. This may cause issues.")
    
    # Check if range combinations file exists, if not, generate it
    if not os.path.exists(range_combinations_file):
        print(f"Range combinations file {range_combinations_file} not found. Generating...")
        generate_range_combinations(splittingPoints_list, range_combinations_file, splitting_points_file, 
                                     edges_index_file)
    
    print(f"Reading range combinations file: {range_combinations_file}")
    
    # Read all range combinations
    all_combinations = []
    with open(range_combinations_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        i, j, k = int(parts[0]), int(parts[1]), int(parts[2])
                        all_combinations.append((i, j, k))
                    except ValueError as e:
                        print(f"Warning: Invalid line {line_num} in {range_combinations_file}: {line.strip()}")
                        print(f"  Error: {e}")
                        continue
                else:
                    print(f"Warning: Invalid line {line_num} in {range_combinations_file}: expected 3 values, got {len(parts)}")
    
    print(f"Total combinations: {len(all_combinations)}")
    
    # Assign tasks to current server (more evenly distributed)
    total_combinations = len(all_combinations)
    
    if total_combinations == 0:
        print(f"No range combinations to process. Server {server_id} exiting.")
        sys.exit(0)
    
    # Handle case when we have fewer combinations than servers
    if total_combinations < total_servers:
        print(f"Warning: Only {total_combinations} combinations but {total_servers} servers.")
        print(f"Only servers 0-{total_combinations-1} will process tasks.")
        if server_id >= total_combinations:
            print(f"Server {server_id} has no tasks to process. Exiting.")
            sys.exit(0)
        # Each server gets at most 1 combination
        my_combinations = [all_combinations[server_id]]
    else:
        # Normal case: distribute combinations evenly
        combinations_per_server = total_combinations // total_servers
        remainder = total_combinations % total_servers
        
        # Distribute remainder evenly: first 'remainder' servers get one extra combination
        if server_id < remainder:
            # This server gets one extra combination
            start_idx = server_id * (combinations_per_server + 1)
            end_idx = start_idx + (combinations_per_server + 1)
        else:
            # This server gets the base number of combinations
            start_idx = remainder * (combinations_per_server + 1) + (server_id - remainder) * combinations_per_server
            end_idx = start_idx + combinations_per_server
        
        my_combinations = all_combinations[start_idx:end_idx]
    
    if len(my_combinations) == 0:
        print(f"Server {server_id} has no tasks to process. Exiting.")
        sys.exit(0)
    
    # Determine range for display
    if total_combinations < total_servers:
        range_str = f"combination {server_id}"
    else:
        range_str = f"combinations {start_idx} to {end_idx-1}"
    
    print(f"Server {server_id} processing {range_str} (total: {len(my_combinations)} combinations)")
    
    # pa is already initialized at module level in create_triangles.py
    # No need to create or set it here
    
    # Process assigned combinations
    index = construct_triangles_for_ranges(my_combinations)
    
    # Filter results
    triangles = filter_index(index)
    
    print(f"Server {server_id}: All triangles={len(index)}, After filter={len(triangles)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to file
    output_file = os.path.join(output_dir, f'triangles_server_{server_id}.csv')
    with open(output_file, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(triangles)
    
    print(f"Results saved to: {output_file}")
    print(f"edgesIndex: {edgesIndex}")
    print(f"--- {time.time() - start_time:.2f} seconds ---")

