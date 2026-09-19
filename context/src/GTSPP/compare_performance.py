#!/usr/bin/env python3
"""
Performance Comparison Script for GTS Optimization
Compares original vs optimized search implementations
"""

import numpy as np
import subprocess
import json
import time
from typing import List, Dict, Tuple

class PerformanceAnalyzer:
    def __init__(self):
        self.results = {
            'original': {},
            'optimized': {}
        }
    
    def parse_output(self, output: str) -> Dict:
        """Parse timing and statistics from program output"""
        data = {}
        
        # Extract search time
        for line in output.split('\n'):
            if 'completed in' in line.lower() or 'time:' in line.lower():
                try:
                    # Extract number before 'ms'
                    parts = line.split('ms')[0].split()
                    data['search_time_ms'] = float(parts[-1])
                except:
                    pass
            
            # Extract pruning statistics
            if 'pruning' in line.lower() and '%' in line:
                try:
                    # Extract pruning rate
                    percentage = line.split('(')[-1].split('%')[0]
                    data['pruning_rate'] = float(percentage)
                except:
                    pass
            
            if 'nodes pruned' in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '/' in part:
                            pruned, total = part.split('/')
                            data['nodes_pruned'] = int(pruned)
                            data['nodes_total'] = int(total)
                except:
                    pass
        
        return data
    
    def compute_recall(self, results_a: np.ndarray, results_b: np.ndarray, k: int) -> float:
        """
        Compute recall@k between two result sets
        results_a: (n_queries, k) - optimized results
        results_b: (n_queries, k) - ground truth results
        """
        if len(results_a) != len(results_b):
            print(f"Warning: result lengths don't match: {len(results_a)} vs {len(results_b)}")
            return 0.0
        
        recalls = []
        for opt, orig in zip(results_a, results_b):
            # Convert to sets and compute intersection
            opt_set = set(opt[:k])
            orig_set = set(orig[:k])
            intersection = len(opt_set & orig_set)
            recalls.append(intersection / k)
        
        return np.mean(recalls)
    
    def analyze_speedup(self, time_original: float, time_optimized: float) -> Dict:
        """Analyze speedup metrics"""
        speedup = time_original / time_optimized if time_optimized > 0 else 0
        
        return {
            'time_original_ms': time_original,
            'time_optimized_ms': time_optimized,
            'speedup': speedup,
            'time_reduction_percent': ((time_original - time_optimized) / time_original * 100) if time_original > 0 else 0
        }
    
    def generate_report(self, output_file: str = 'performance_report.json'):
        """Generate comprehensive performance report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'comparison': self.results,
            'summary': {}
        }
        
        # Calculate summary statistics
        if 'search_time_ms' in self.results['original'] and 'search_time_ms' in self.results['optimized']:
            speedup_data = self.analyze_speedup(
                self.results['original']['search_time_ms'],
                self.results['optimized']['search_time_ms']
            )
            report['summary'].update(speedup_data)
        
        # Add pruning comparison
        if 'pruning_rate' in self.results['original'] and 'pruning_rate' in self.results['optimized']:
            report['summary']['pruning_improvement'] = (
                self.results['optimized']['pruning_rate'] - self.results['original']['pruning_rate']
            )
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_file}")
        return report
    
    def print_summary(self):
        """Print human-readable summary"""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*60)
        
        if 'search_time_ms' in self.results['original'] and 'search_time_ms' in self.results['optimized']:
            orig_time = self.results['original']['search_time_ms']
            opt_time = self.results['optimized']['search_time_ms']
            speedup = orig_time / opt_time if opt_time > 0 else 0
            
            print(f"\nSearch Time:")
            print(f"  Original:  {orig_time:.2f} ms")
            print(f"  Optimized: {opt_time:.2f} ms")
            print(f"  Speedup:   {speedup:.2f}x")
            
            if speedup >= 5.0:
                print(f"  ✓ Target achieved (5-10x)!")
            elif speedup >= 3.0:
                print(f"  ⚠ Good but below target (got {speedup:.1f}x, target 5-10x)")
            else:
                print(f"  ✗ Below target (got {speedup:.1f}x, target 5-10x)")
        
        if 'pruning_rate' in self.results['original'] and 'pruning_rate' in self.results['optimized']:
            orig_prune = self.results['original']['pruning_rate']
            opt_prune = self.results['optimized']['pruning_rate']
            
            print(f"\nPruning Rate:")
            print(f"  Original:  {orig_prune:.2f}%")
            print(f"  Optimized: {opt_prune:.2f}%")
            print(f"  Improvement: +{opt_prune - orig_prune:.2f}%")
        
        if 'recall' in self.results.get('comparison', {}):
            recall = self.results['comparison']['recall']
            print(f"\nRecall@k: {recall:.4f}")
            if recall >= 0.95:
                print(f"  ✓ Maintained (≥0.95)")
            elif recall >= 0.90:
                print(f"  ⚠ Slightly decreased (0.90-0.95)")
            else:
                print(f"  ✗ Significantly decreased (<0.90)")
        
        print("\n" + "="*60)


def load_results_from_file(filename: str) -> np.ndarray:
    """Load search results from file"""
    try:
        # Assuming results are saved as text file with one result per line
        with open(filename, 'r') as f:
            lines = f.readlines()
            results = []
            for line in lines:
                # Parse line: expected format "qid: id1 id2 id3 ..."
                if ':' in line:
                    ids = line.split(':')[1].strip().split()
                    results.append([int(x) for x in ids])
                else:
                    ids = line.strip().split()
                    results.append([int(x) for x in ids])
            return np.array(results)
    except Exception as e:
        print(f"Error loading results from {filename}: {e}")
        return np.array([])


def main():
    analyzer = PerformanceAnalyzer()
    
    print("GTS Performance Comparison Tool")
    print("="*60)
    
    # Option 1: Parse from existing log files
    print("\nOption 1: Parse from log files")
    print("Place your program outputs in:")
    print("  - original_output.txt")
    print("  - optimized_output.txt")
    
    try:
        with open('original_output.txt', 'r') as f:
            original_output = f.read()
            analyzer.results['original'] = analyzer.parse_output(original_output)
            print("✓ Loaded original output")
    except FileNotFoundError:
        print("⚠ original_output.txt not found")
    
    try:
        with open('optimized_output.txt', 'r') as f:
            optimized_output = f.read()
            analyzer.results['optimized'] = analyzer.parse_output(optimized_output)
            print("✓ Loaded optimized output")
    except FileNotFoundError:
        print("⚠ optimized_output.txt not found")
    
    # Option 2: Compare result files
    print("\nOption 2: Compare result files for recall computation")
    print("Place your result files:")
    print("  - original_results.txt")
    print("  - optimized_results.txt")
    
    try:
        original_results = load_results_from_file('original_results.txt')
        optimized_results = load_results_from_file('optimized_results.txt')
        
        if len(original_results) > 0 and len(optimized_results) > 0:
            k = min(10, original_results.shape[1] if len(original_results.shape) > 1 else 10)
            recall = analyzer.compute_recall(optimized_results, original_results, k)
            analyzer.results['comparison'] = {'recall': recall}
            print(f"✓ Computed recall@{k}: {recall:.4f}")
    except Exception as e:
        print(f"⚠ Could not compute recall: {e}")
    
    # Generate report
    if analyzer.results['original'] or analyzer.results['optimized']:
        analyzer.print_summary()
        analyzer.generate_report()
    else:
        print("\n⚠ No data to analyze. Please provide log or result files.")
        print("\nUsage:")
        print("1. Run your original implementation and save output to 'original_output.txt'")
        print("2. Run your optimized implementation and save output to 'optimized_output.txt'")
        print("3. Run this script to see the comparison")
        print("\nExample:")
        print("  ./original_program > original_output.txt 2>&1")
        print("  ./optimized_program > optimized_output.txt 2>&1")
        print("  python3 compare_performance.py")


if __name__ == '__main__':
    main()
