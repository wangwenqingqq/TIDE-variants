import sys
import re

def parse_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    
    data = {}
    
    # Parse Result radius
    # It seems the numbers start after "Result radius: \n"
    # We look for the line "Result radius:" and take the content after it until the next label or end of file
    radius_match = re.search(r'Result radius:\s*\n([\d\s\.]+)', content)
    if radius_match:
        radius_str = radius_match.group(1)
        # Split by whitespace and convert to float
        data['radius'] = [float(x) for x in radius_str.strip().split()]
    else:
        data['radius'] = []

    # Parse Time of index construction
    time_index_match = re.search(r'Time of index construction:\s*([\d\.]+)', content)
    if time_index_match:
        data['time_index'] = float(time_index_match.group(1))
    else:
        data['time_index'] = None

    # Parse Average search time
    time_search_match = re.search(r'Average search time:\s*([\d\.]+)', content)
    if time_search_match:
        data['time_search'] = float(time_search_match.group(1))
    else:
        data['time_search'] = None
        
    return data

def compare_files(file1, file2):
    data1 = parse_file(file1)
    data2 = parse_file(file2)
    
    if data1 is None or data2 is None:
        return

    print(f"Comparing {file1} vs {file2}")
    print("-" * 50)
    
    # Compare Result radius
    r1 = data1['radius']
    r2 = data2['radius']
    
    if len(r1) != len(r2):
        print(f"Warning: Number of results differ! ({len(r1)} vs {len(r2)})")
    
    min_len = min(len(r1), len(r2))
    matches = 0
    for i in range(min_len):
        # Using a small epsilon for float comparison
        if abs(r1[i] - r2[i]) < 1e-4:
            matches += 1
            
    print(f"Result radius matching quantity: {matches} / {min_len}")
    if min_len > 0:
        print(f"Match rate: {matches/min_len*100:.2f}%")
    
    # Compare Time of index construction
    t1_idx = data1['time_index']
    t2_idx = data2['time_index']
    print(f"\nTime of index construction:")
    print(f"  File 1: {t1_idx}")
    print(f"  File 2: {t2_idx}")
    if t1_idx is not None and t2_idx is not None:
        diff = t2_idx - t1_idx
        print(f"  Diff: {diff:+.6f} s")

    # Compare Average search time
    t1_search = data1['time_search']
    t2_search = data2['time_search']
    print(f"\nAverage search time:")
    print(f"  File 1: {t1_search}")
    print(f"  File 2: {t2_search}")
    if t1_search is not None and t2_search is not None:
        diff = t2_search - t1_search
        print(f"  Diff: {diff:+.6f} s")
        if t1_search > 0:
            speedup = (t1_search - t2_search) / t1_search * 100
            print(f"  Improvement: {speedup:+.2f}% (positive means faster)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default for user convenience if running without args in this context
        file1 = "cost_q_k_hei_ori.txt"
        file2 = "cost_q_k_lea_prune86.txt"
        print(f"No arguments provided. Using default: {file1} {file2}\n")
        compare_files(file1, file2)
    else:
        compare_files(sys.argv[1], sys.argv[2])
