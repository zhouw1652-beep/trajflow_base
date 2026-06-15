"""
evaluate_mirage_5d.py

Compute 5D JSD metrics for MIRAGE generated trajectories using the official
src.out_evaluation.statistical_metrics.get_five_official_metrics.

Dimensions:
    1. Distance    -- per-trajectory total consecutive Haversine distance (km)
    2. Radius     -- per-trajectory gyration radius (km)
    3. DailyLoc   -- per-trajectory unique checkins count
    4. Interval   -- global interval distribution (diff of [t_start] + arrival_times + [t_end])
    5. Category   -- global category distribution (marks, across all sequences)

Usage:
    python scripts/evaluate_mirage_5d.py --city NewYork
    python scripts/evaluate_mirage_5d.py --city Tokyo --allow_truncate_debug

Output:
    outputs/mirage_baseline/{city}/metrics_5d.json
    outputs/mirage_baseline/{city}/metrics_5d.txt
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
TRAJFLOW_ROOT = (SCRIPT_DIR / "..").resolve()

# Add TrajFlow root to sys.path so we can import statistical_metrics
if str(TRAJFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAJFLOW_ROOT))

from src.out_evaluation.statistical_metrics import get_five_official_metrics
from scripts.mirage_io import load_mirage_pkl


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIRAGE 5D JSD metrics")
    parser.add_argument('--city', type=str, required=True,
                       choices=['NewYork', 'Tokyo', 'Istanbul'])
    parser.add_argument('--min_seq_len', type=int, default=1,
                       help='Minimum sequence length threshold (default: 1)')
    parser.add_argument('--allow_truncate_debug', action='store_true', default=False,
                       help='Allow truncating real/gen to min length on count mismatch. '
                            'Default: False (errors on mismatch).')
    args = parser.parse_args()
    city = args.city
    min_seq_len = args.min_seq_len
    allow_truncate = args.allow_truncate_debug

    base_dir = TRAJFLOW_ROOT / "outputs" / "mirage_baseline" / city
    output_json = base_dir / "metrics_5d.json"
    output_txt  = base_dir / "metrics_5d.txt"

    print(f"\n{'='*60}")
    print(f"MIRAGE 5D Evaluation -- {city}")
    print(f"{'='*60}\n")

    # ── Load real sequences (via common mirage_io loader) ──
    mirage_obj = load_mirage_pkl(city, TRAJFLOW_ROOT)
    real_all = mirage_obj['sequences']
    print(f"  Loaded real sequences: {len(real_all)} total")

    # ── Filter to test split ──
    mirage_dir = TRAJFLOW_ROOT / "data" / "mirage_trajflow" / city
    split_test_path = mirage_dir / "split_test.pkl"
    if not split_test_path.exists():
        raise FileNotFoundError(
            f"split_test.pkl not found at {split_test_path}. "
            "Cannot proceed without official test split for evaluation."
        )
    with open(split_test_path, 'rb') as f:
        test_indices = pickle.load(f)
    real_seqs = [real_all[i] for i in test_indices]
    print(f"  Filtered to test split: {len(real_seqs)} sequences (from split_test.pkl)")

    # ── Load generated sequences ──
    gen_pkl_path = base_dir / "generated.pkl"
    if not gen_pkl_path.exists():
        raise FileNotFoundError(
            f"generated.pkl not found at {gen_pkl_path}.\n"
            f"Run postprocess first: python scripts/postprocess_mirage_trajflow_generated.py --city {city}"
        )
    with open(gen_pkl_path, 'rb') as f:
        gen_obj = pickle.load(f)
    gen_seqs = gen_obj['sequences']
    print(f"  Loaded generated: {len(gen_seqs)} sequences")

    # ── P0-4: Count mismatch -> error (unless --allow_truncate_debug) ──
    print(f"\n  Real test:   {len(real_seqs)} sequences")
    print(f"  Generated:   {len(gen_seqs)} sequences")
    if len(real_seqs) != len(gen_seqs):
        msg = (
            f"Sequence count mismatch: real={len(real_seqs)}, gen={len(gen_seqs)}. "
            "Do not truncate for official evaluation."
        )
        if allow_truncate:
            print(f"\n  WARNING: {msg}")
            print(f"  (--allow_truncate_debug is set; truncating to min length)")
            min_n = min(len(real_seqs), len(gen_seqs))
            real_seqs = real_seqs[:min_n]
            gen_seqs  = gen_seqs[:min_n]
        else:
            raise ValueError(msg)

    # ── Compute 5D official metrics ──
    print(f"\n  Computing 5D JSD metrics (via get_five_official_metrics)...")
    metrics = get_five_official_metrics(real_seqs, gen_seqs, min_seq_len=min_seq_len)

    # Add metadata
    metrics['n_sequences'] = len(gen_seqs)
    metrics['city'] = city
    metrics['min_seq_len'] = min_seq_len

    # ── Print results ──
    print(f"\n{'='*50}")
    print(f"MIRAGE 5D JSD Metrics -- {city}")
    print(f"{'='*50}")
    print(f"  Distance JSD:  {metrics['distance_jsd']:.6f}")
    print(f"  Radius JSD:    {metrics['radius_jsd']:.6f}")
    print(f"  DailyLoc JSD:  {metrics['dailyloc_jsd']:.6f}")
    print(f"  Interval JSD:  {metrics['interval_jsd']:.6f}")
    print(f"  Category JSD:  {metrics['category_jsd']:.6f}")
    print(f"  ─────────────────────────────")
    print(f"  Mean JSD (5D): {metrics['mean_jsd']:.6f}")
    print(f"{'='*50}")

    # ── Write outputs ──
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(output_txt, 'w') as f:
        f.write(f"MIRAGE 5D Evaluation -- {city}\n")
        f.write(f"{'='*40}\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write(f"{'='*40}\n")

    print(f"\n  Written: {output_json}")
    print(f"  Written: {output_txt}")

    print(f"\n{'='*60}")
    print(f"Evaluation complete: {city}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
