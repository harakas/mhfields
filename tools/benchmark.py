#!/usr/bin/env python3
"""
Benchmark script for magnetic field performance measurement.

Usage:
    python tools/benchmark.py
"""

import numpy as np
import time

from mhfields import fields
from mhfields.fields import ring_magnetic_field, ring_magnetic_field_original


def benchmark_function(func, r_grid, a_grid, R, I, n_runs=5):
    """Run function multiple times and return average time."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(r_grid, a_grid, R, I)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def run_benchmark(grid_size, R=1.0, I=1.0, n_runs=5):
    """Run benchmark for a given grid size."""
    r = np.linspace(0, 3, grid_size)
    a = np.linspace(-3, 3, grid_size)
    r_grid, a_grid = np.meshgrid(r, a)

    n_points = grid_size * grid_size

    # Warmup round (JIT, cache warming, etc.)
    ring_magnetic_field_original(r_grid, a_grid, R, I)
    ring_magnetic_field(r_grid, a_grid, R, I)

    # Benchmark original
    t_orig, std_orig = benchmark_function(
        ring_magnetic_field_original, r_grid, a_grid, R, I, n_runs
    )

    # Benchmark current
    t_curr, std_curr = benchmark_function(
        ring_magnetic_field, r_grid, a_grid, R, I, n_runs
    )

    # Compute speedup
    speedup = t_orig / t_curr if t_curr > 0 else float('inf')

    return {
        'grid_size': grid_size,
        'n_points': n_points,
        't_original': t_orig,
        'std_original': std_orig,
        't_current': t_curr,
        'std_current': std_curr,
        'speedup': speedup,
    }


def verify_correctness(grid_size=100, R=1.0, I=1.0):
    """Verify that both functions produce identical results."""
    r = np.linspace(0, 3, grid_size)
    a = np.linspace(-3, 3, grid_size)
    r_grid, a_grid = np.meshgrid(r, a)

    B_r_orig, B_a_orig = ring_magnetic_field_original(r_grid, a_grid, R, I)
    B_r_curr, B_a_curr = ring_magnetic_field(r_grid, a_grid, R, I)

    max_diff_r = np.max(np.abs(B_r_orig - B_r_curr))
    max_diff_a = np.max(np.abs(B_a_orig - B_a_curr))

    return max_diff_r, max_diff_a


def main():
    print("=" * 70)
    print("Magnetic Field Benchmark")
    print("=" * 70)

    # Verify correctness first
    print("\nVerifying correctness...")
    max_diff_r, max_diff_a = verify_correctness()
    print(f"  Max difference B_r: {max_diff_r:.2e}")
    print(f"  Max difference B_a: {max_diff_a:.2e}")
    if max_diff_r < 1e-15 and max_diff_a < 1e-15:
        print("  OK - Results are identical")
    else:
        print("  WARNING - Results differ!")

    # Run benchmarks
    print("\nRunning benchmarks (5 runs each)...")
    print("-" * 70)
    print(f"{'Grid':<10} {'Points':<12} {'Original':<14} {'Current':<14} {'Speedup':<10}")
    print("-" * 70)

    grid_sizes = [100, 200, 500, 1000]

    for size in grid_sizes:
        result = run_benchmark(size, n_runs=5)
        print(
            f"{size}x{size:<6} {result['n_points']:<12} "
            f"{result['t_original']*1000:>8.2f} ms    "
            f"{result['t_current']*1000:>8.2f} ms    "
            f"{result['speedup']:>6.2f}x"
        )

    print("-" * 70)
    print()


def run_timing_analysis(grid_size=500, R=1.0, I=1.0, n_runs=3):
    """Run timing analysis to identify bottlenecks."""
    print("=" * 70)
    print("Timing Analysis")
    print("=" * 70)

    r = np.linspace(0, 3, grid_size)
    a = np.linspace(-3, 3, grid_size)
    r_grid, a_grid = np.meshgrid(r, a)

    # Enable timing debug
    fields._TIMING_DEBUG = True

    # Warmup
    ring_magnetic_field(r_grid, a_grid, R, I)

    # Collect stats from multiple runs
    all_stats = []
    for _ in range(n_runs):
        ring_magnetic_field(r_grid, a_grid, R, I)
        all_stats.append(dict(fields._timing_stats))

    # Disable timing debug
    fields._TIMING_DEBUG = False

    # Average the stats
    avg_stats = {}
    for key in all_stats[0].keys():
        avg_stats[key] = np.mean([s[key] for s in all_stats])

    print(f"\nGrid size: {grid_size}x{grid_size} = {grid_size*grid_size} points")
    print(f"Average of {n_runs} runs (after warmup)")
    print(f"\nTiming breakdown:")
    print("-" * 50)

    total = avg_stats.get('total', 0)
    for key, value in sorted(avg_stats.items()):
        if key != 'total':
            pct = (value / total * 100) if total > 0 else 0
            print(f"  {key:<25} {value*1000:>8.2f} ms  ({pct:>5.1f}%)")

    print("-" * 50)
    print(f"  {'total':<25} {total*1000:>8.2f} ms  (100.0%)")
    print()


if __name__ == '__main__':
    main()
    run_timing_analysis(grid_size=2000, n_runs=3)
