"""
analyze_eval_coverage.py

Reads a FlingBot `replay_buffer.hdf5` (produced by `run_sim.py --eval`)
and extracts, per task/episode:
    - initial (preaction) coverage
    - final (postaction) coverage
    - delta coverage (final - initial)
    - normalized final coverage (final / max_coverage), if max_coverage
      is available in this file

...and aggregate statistics (mean/median/std) across the whole dataset.

Usage:
    python analyze_eval_coverage.py flingbot_eval_X/replay_buffer.hdf5
    python analyze_eval_coverage.py flingbot_eval_X/replay_buffer.hdf5 --csv out.csv

This script is intentionally defensive about the exact on-disk layout:
each "episode" in the file may store per-step values (preaction_coverage,
postaction_coverage, etc.) either as HDF5 *datasets* (arrays, one entry
per step) or as HDF5 *attrs* (scalars, e.g. if only ever written once).
It checks both, and skips/reports anything it can't find rather than
guessing incorrect numbers.
"""
import argparse
import csv
import sys
import statistics
import h5py
import numpy as np


def _get_value(group, key):
    """
    Try to fetch `key` from an h5py group, checking both attrs and
    datasets, and both scalar and array storage. Returns:
        - a 1-D numpy array of values if found as a dataset
        - a 1-element list [value] if found as a scalar attr
        - None if not found anywhere
    """
    if key in group.attrs:
        return np.atleast_1d(group.attrs[key])
    if key in group:
        data = group[key][()]
        return np.atleast_1d(data)
    return None


def extract_task_stats(group, task_name):
    """
    Pull preaction/postaction/max coverage for a single episode group.
    Returns a dict of stats, or None if this episode has no usable
    coverage data (e.g. a "no valid action found" episode that we
    intentionally chose not to write anything for).
    """
    preaction = _get_value(group, 'preaction_coverage')
    postaction = _get_value(group, 'postaction_coverage')

    if preaction is None or postaction is None or \
            len(preaction) == 0 or len(postaction) == 0:
        return None

    initial_coverage = float(preaction[0])
    final_coverage = float(postaction[-1])
    delta_coverage = final_coverage - initial_coverage

    max_coverage = _get_value(group, 'max_coverage')
    normalized_final = None
    if max_coverage is not None and len(max_coverage) > 0 \
            and float(max_coverage[0]) > 0:
        normalized_final = final_coverage / float(max_coverage[0])

    num_steps = max(len(preaction), len(postaction))

    cloth_stuck = bool(group.attrs.get('cloth_stuck', False)) \
        if 'cloth_stuck' in group.attrs else False
    timed_out = bool(group.attrs.get('timed_out', False)) \
        if 'timed_out' in group.attrs else False

    return {
        'task': task_name,
        'num_steps': num_steps,
        'initial_coverage': initial_coverage,
        'final_coverage': final_coverage,
        'delta_coverage': delta_coverage,
        'max_coverage': float(max_coverage[0]) if max_coverage is not None else None,
        'normalized_final_coverage': normalized_final,
        'cloth_stuck': cloth_stuck,
        'timed_out': timed_out,
    }


def analyze(hdf5_path, csv_path=None):
    per_task = []
    skipped = []

    with h5py.File(hdf5_path, 'r') as f:
        # Each top-level key is expected to be one episode/task group.
        task_names = list(f.keys())
        print(f"Found {len(task_names)} episode group(s) in {hdf5_path}\n")

        for task_name in task_names:
            group = f[task_name]
            if not isinstance(group, h5py.Group):
                continue
            stats = extract_task_stats(group, task_name)
            if stats is None:
                skipped.append(task_name)
                continue
            per_task.append(stats)

    if not per_task:
        print("No usable episodes found (no preaction/postaction "
              "coverage data). Nothing to report.")
        if skipped:
            print(f"Skipped {len(skipped)} empty/invalid group(s): "
                  f"{skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
        return

    # ---- Per-task report ----
    print(f"{'Task':<20}{'Steps':>6}{'Initial':>10}{'Final':>10}"
          f"{'Delta':>10}{'Final/Max':>12}")
    print("-" * 68)
    for s in per_task:
        norm_str = f"{s['normalized_final_coverage']:.3f}" \
            if s['normalized_final_coverage'] is not None else "n/a"
        flags = []
        if s['cloth_stuck']:
            flags.append("STUCK")
        if s['timed_out']:
            flags.append("TIMEOUT")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{s['task']:<20}{s['num_steps']:>6}"
              f"{s['initial_coverage']:>10.3f}{s['final_coverage']:>10.3f}"
              f"{s['delta_coverage']:>10.3f}{norm_str:>12}{flag_str}")

    # ---- Aggregate stats ----
    deltas = [s['delta_coverage'] for s in per_task]
    finals = [s['final_coverage'] for s in per_task]
    normalized_finals = [s['normalized_final_coverage'] for s in per_task
                          if s['normalized_final_coverage'] is not None]
    n_stuck = sum(1 for s in per_task if s['cloth_stuck'])
    n_timeout = sum(1 for s in per_task if s['timed_out'])

    def _fmt(vals):
        if not vals:
            return "n/a"
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        median = statistics.median(vals)
        return f"mean={mean:.4f}  median={median:.4f}  std={std:.4f}  n={len(vals)}"

    print("\n" + "=" * 68)
    print("AGGREGATE STATS")
    print("=" * 68)
    print(f"Tasks evaluated:        {len(per_task)}")
    if skipped:
        print(f"Tasks skipped (no data): {len(skipped)}")
    print(f"Tasks with cloth stuck:  {n_stuck}")
    print(f"Tasks timed out:         {n_timeout}")
    print(f"Delta coverage:          {_fmt(deltas)}")
    print(f"Final coverage:          {_fmt(finals)}")
    if normalized_finals:
        print(f"Final / max coverage:   {_fmt(normalized_finals)}")
    else:
        print("Final / max coverage:   n/a (no 'max_coverage' field "
              "found in this replay buffer)")

    # ---- Optional CSV export ----
    if csv_path:
        fieldnames = ['task', 'num_steps', 'initial_coverage',
                      'final_coverage', 'delta_coverage', 'max_coverage',
                      'normalized_final_coverage', 'cloth_stuck', 'timed_out']
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in per_task:
                writer.writerow(s)
        print(f"\nPer-task data written to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract final/delta coverage stats from a FlingBot "
                     "eval replay_buffer.hdf5")
    parser.add_argument('hdf5_path', help="Path to replay_buffer.hdf5")
    parser.add_argument('--csv', default=None,
                         help="Optional path to write a per-task CSV")
    args = parser.parse_args()
    analyze(args.hdf5_path, csv_path=args.csv)
