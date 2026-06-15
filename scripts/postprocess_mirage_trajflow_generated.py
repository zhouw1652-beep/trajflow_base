"""
postprocess_mirage_trajflow_generated.py

Convert generated_trajectories.csv -> MIRAGE-format generated.pkl.

Usage:
    python scripts/postprocess_mirage_trajflow_generated.py --city NewYork
    python scripts/postprocess_mirage_trajflow_generated.py --city NewYork --csv outputs/.../generated_trajectories.csv
    python scripts/postprocess_mirage_trajflow_generated.py --city NewYork --time_mode csv_time

Output:
    outputs/mirage_baseline/{city}/generated.pkl

Logic:
    1. glob latest CSV under outputs/mirage_baseline/{city}/generation/**
    2. Load test_metadata.pkl and test_indices.pkl
    3. Verify CSV uid matches selected_indices.json:
         selected_indices[uid] == test_metadata[uid]["dataset_index"]
         Mismatch raises ValueError.
    4. Length-align: uniformly sample L points from 120-gen points
    5. KDTree project GPS -> POI IDs
    6. Rebuild MIRAGE fields: marks, checkins, revisit, arrival_times, day_hour
    7. GPS sanity check: per-trajectory out-of-bounds count -> overall summary
       STOP only if >5% of all points out of bounds or >20% of trajectories affected
    8. GPS->POI Haversine distance statistics
"""

import argparse
import glob
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).parent.resolve()
TRAJFLOW_ROOT = (SCRIPT_DIR / "..").resolve()
if str(TRAJFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAJFLOW_ROOT))

from scripts.mirage_io import load_mirage_pkl


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_gps_coord(coord):
    """Parse a GPS coordinate from str/list/tuple into (lat, lon) float tuple."""
    if isinstance(coord, str):
        parts = coord.split(',')
        return float(parts[0]), float(parts[1])
    elif isinstance(coord, (list, tuple)):
        return float(coord[0]), float(coord[1])
    else:
        raise ValueError(f"Cannot parse GPS coord of type {type(coord)}: {coord}")


def find_csv(base_dir, explicit_csv=None):
    if explicit_csv:
        if not os.path.exists(explicit_csv):
            raise FileNotFoundError(f"Explicit CSV not found: {explicit_csv}")
        return explicit_csv

    pattern = os.path.join(base_dir, "generation", "**", "generated_trajectories.csv")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No generated_trajectories.csv found under {base_dir}/generation/\n"
            f"  Pattern: {pattern}\n"
            f"  Did you run generate.py first?"
        )

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = candidates[0]
    print(f"\n  Found {len(candidates)} CSV candidates:")
    for c in candidates[:5]:
        print(f"    {c}")
    print(f"  Using: {chosen}")
    return chosen


def load_selected_indices(csv_dir):
    si_path = os.path.join(csv_dir, "selected_indices.json")
    if os.path.exists(si_path):
        with open(si_path) as f:
            d = json.load(f)
        arr = np.array(d['selected_indices'], dtype=int)
        print(f"  Loaded selected_indices from {si_path}: {len(arr)} entries")
        return arr
    print(f"  WARNING: selected_indices.json not found in {csv_dir}")
    return None


def uniform_sample(n_items, L):
    if L <= 0:
        return []
    if L >= n_items:
        return list(range(n_items))
    return [int(round(i)) for i in np.linspace(0, n_items - 1, L)]


def check_gps_bounds_batch(traj_gps_lists, lat_min, lat_max, lon_min, lon_max,
                            padding_deg=0.05):
    """
    Check GPS bounds for all trajectories. Collects out-of-bounds info.
    Returns (total_oob, total_points, trajs_affected, traj_oob_counts).
    Raises ValueError only if the overall situation is clearly abnormal.
    """
    total_oob = 0
    total_points = 0
    trajs_affected = 0
    traj_oob_counts = []
    severe_trajs = []

    for ti, gps_list in enumerate(traj_gps_lists):
        oob_this = 0
        for lat, lon in gps_list:
            total_points += 1
            if not (lat_min - padding_deg <= lat <= lat_max + padding_deg and
                    lon_min - padding_deg <= lon <= lon_max + padding_deg):
                total_oob += 1
                oob_this += 1
        traj_oob_counts.append(oob_this)
        if oob_this > 0:
            trajs_affected += 1
            if oob_this > len(gps_list) * 0.5:
                severe_trajs.append(ti)

    print(f"\n  GPS Bounds Check Summary:")
    print(f"    Total points checked: {total_points}")
    print(f"    Out-of-bounds points: {total_oob} ({100*total_oob/max(total_points,1):.2f}%)")
    print(f"    Trajectories affected: {trajs_affected}/{len(traj_gps_lists)}")
    if severe_trajs:
        print(f"    Severe (>50% points OOB): {len(severe_trajs)} trajectories")
        print(f"      Indices: {severe_trajs[:10]}{'...' if len(severe_trajs)>10 else ''}")

    oob_ratio = total_oob / max(total_points, 1)
    traj_ratio = trajs_affected / max(len(traj_gps_lists), 1)

    # STOP only if clearly abnormal
    if oob_ratio > 0.05 or traj_ratio > 0.20:
        raise ValueError(
            f"GPS bounds check FAILED: {total_oob}/{total_points} points ({100*oob_ratio:.1f}%) "
            f"out of bounds, {trajs_affected}/{len(traj_gps_lists)} trajectories affected. "
            f"This indicates a serious denormalization or model issue. Stop."
        )
    elif oob_ratio > 0.01 or traj_ratio > 0.05:
        print(f"  NOTE: Some GPS points are out of bounds (mild). Continuing...")
    else:
        print(f"  GPS bounds check: PASSED")


def get_poi_category(cat_dict, key):
    """Look up category by POI key, trying original, int, and str forms."""
    candidates = [key]
    try:
        candidates.append(int(key))
    except (TypeError, ValueError):
        pass
    candidates.append(str(key))

    for k in candidates:
        if k in cat_dict:
            return cat_dict[k]

    raise KeyError(
        f"POI id {key!r} not found in poi_category. "
        f"Sample keys: {list(cat_dict.keys())[:10]}"
    )


def postprocess(city, explicit_csv=None, time_mode='scale_total_time', skip_gps_check=False):
    base_dir = TRAJFLOW_ROOT / "outputs" / "mirage_baseline" / city
    output_pkl = base_dir / "generated.pkl"

    print(f"\n{'='*60}")
    print(f"Postprocessing: {city}")
    print(f"  time_mode = {time_mode}")
    print(f"{'='*60}\n")

    # 1. Find CSV
    print("[1/7] Locating generated CSV...")
    csv_path = find_csv(str(base_dir), explicit_csv)
    csv_dir = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)
    print(f"  Loaded CSV: {len(df)} rows, columns: {list(df.columns)}")
    n_traj = df['uid'].nunique()
    print(f"  Unique trajectories: {n_traj}")

    # 2. Load metadata
    print("\n[2/7] Loading test metadata...")
    mirage_dir = TRAJFLOW_ROOT / "data" / "mirage_trajflow" / city
    test_meta_path = mirage_dir / "test_metadata.pkl"
    test_indices_path = mirage_dir / "test_indices.pkl"
    with open(test_meta_path, 'rb') as f:
        test_metadata = pickle.load(f)
    with open(test_indices_path, 'rb') as f:
        test_indices = pickle.load(f)
    print(f"  test_metadata: {len(test_metadata)} items")
    print(f"  test_indices: {len(test_indices)} items")

    # Load original MIRAGE pkl for poi_gps / poi_category
    mirage_obj = load_mirage_pkl(city, TRAJFLOW_ROOT)
    poi_gps = mirage_obj['poi_gps']
    poi_cat = mirage_obj['poi_category']
    print(f"  poi_gps: {len(poi_gps)} POIs")
    print(f"  poi_category: {len(poi_cat)} categories")

    # 3. STRICT index alignment check (P0-5)
    # P0-3: CSV uid count check BEFORE selected_indices alignment check
    csv_uids = sorted(df['uid'].unique())
    expected_count = len(test_metadata)

    if len(csv_uids) != expected_count:
        raise ValueError(
            f"CSV uid count mismatch: csv={len(csv_uids)}, expected={expected_count}. "
            "Generation did not produce exactly one trajectory per test sequence."
        )

    if csv_uids != list(range(expected_count)):
        raise ValueError(
            f"CSV uid values must be contiguous 0..{expected_count-1}, "
            f"got first={csv_uids[0]}, last={csv_uids[-1]}."
        )

    print(f"  CSV uid count check: PASSED ({expected_count} trajectories)")

    print("\n[3/7] Verifying STRICT index alignment...")
    selected_indices = load_selected_indices(csv_dir)

    if selected_indices is not None:
        if len(selected_indices) != expected_count:
            raise ValueError(
                f"selected_indices.json has {len(selected_indices)} entries "
                f"but test_metadata has {expected_count}. Stop."
            )
        # P0-5: strict check -- selected_indices[uid] must equal test_metadata[uid]["dataset_index"]
        for uid in csv_uids:
            sel_idx = int(selected_indices[uid])
            meta_dataset_idx = int(test_metadata[uid]['dataset_index'])
            if sel_idx != meta_dataset_idx:
                raise ValueError(
                    f"Index mismatch at uid={uid}: "
                    f"selected_indices[{uid}]={sel_idx} != "
                    f"test_metadata[{uid}]['dataset_index']={meta_dataset_idx}. Stop."
                )
        print(f"  STRICT alignment verified: selected_indices[uid] == test_metadata[uid]['dataset_index'] for all uid OK")
    else:
        raise FileNotFoundError(
            f"selected_indices.json not found in {csv_dir}. "
            f"Cannot verify alignment. Stop. Please ensure generate.py saves selected_indices.json."
        )

    # 4. Group CSV rows by uid
    print("\n[4/7] Grouping trajectories...")
    groups = {uid: grp for uid, grp in df.groupby('uid')}
    all_uids = sorted(groups.keys())
    assert list(all_uids) == list(range(len(all_uids))), "UIDs must be contiguous 0..N-1"
    print(f"  {len(groups)} trajectories grouped")

    # 5. Build POI KDTree
    print("\n[5/7] Building POI KDTree...")
    # poi_gps keys may be int or str; sort numerically when possible
    def sort_key(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return x
    poi_ids = sorted(poi_gps.keys(), key=sort_key)
    valid_categories = set(poi_cat.values())
    print(f"  Valid category set: {len(valid_categories)} unique categories")
    poi_coords = np.array([
        parse_gps_coord(poi_gps[pid])
        for pid in poi_ids
    ], dtype=np.float64)
    tree = cKDTree(poi_coords)
    lat_min = float(poi_coords[:, 0].min())
    lat_max = float(poi_coords[:, 0].max())
    lon_min = float(poi_coords[:, 1].min())
    lon_max = float(poi_coords[:, 1].max())
    print(f"  City bounds: lat=[{lat_min:.4f}, {lat_max:.4f}], lon=[{lon_min:.4f}, {lon_max:.4f}]")
    print(f"  POI IDs: {len(poi_ids)} unique")

    # 6. Process each trajectory
    print("\n[6/7] Processing trajectories...")
    generated_sequences = []
    all_gps_distances = []

    for i, uid in enumerate(all_uids):
        grp = groups[uid].reset_index(drop=True)
        orig_len = test_metadata[i]['original_length']

        # GPS points from CSV (all 120)
        gen_gps_list = [
            [float(grp.iloc[j]['latitude']), float(grp.iloc[j]['longitude'])]
            for j in range(len(grp))
        ]
        gen_arr = np.array(gen_gps_list, dtype=np.float64)

        # Uniform sample L points from 120
        if len(gen_gps_list) != 120:
            raise ValueError(
                f"Trajectory {uid} has {len(gen_gps_list)} points, expected 120. "
                f"Check TrajFlow generate.py trajectory_length config."
            )
        sampled_idx = uniform_sample(120, orig_len)
        sampled_gps = [gen_gps_list[j] for j in sampled_idx]
        sampled_arr = gen_arr[sampled_idx]

        # KDTree projection
        _, nearest_pos = tree.query(sampled_arr)
        checkins = [int(poi_ids[pos]) for pos in nearest_pos]

        # Haversine distances GPS->nearest POI
        nearest_coords = poi_coords[nearest_pos]
        lat1 = sampled_arr[:, 0]
        lon1 = sampled_arr[:, 1]
        lat2 = nearest_coords[:, 0]
        lon2 = nearest_coords[:, 1]
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        dists_km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        all_gps_distances.extend(dists_km.tolist())

        # POI category (P1-2: strict, no default 0)
        marks = [get_poi_category(poi_cat, c) for c in checkins]

        # Revisit
        revisit = [0] * len(checkins)
        seen = set()
        for j, c in enumerate(checkins):
            if c in seen:
                revisit[j] = 1
            seen.add(c)

        # Arrival times and day_hour
        if time_mode == 'csv_time':
            dep_ts = pd.to_datetime(grp.iloc[0]['departure_time'])
            arrival_times = []
            for j in sampled_idx:
                ts = pd.to_datetime(grp.iloc[j]['time'])
                delta = (ts - dep_ts).total_seconds()
                arrival_times.append(delta / 86400.0)
            t_start = 0.0
            t_end = arrival_times[-1] if arrival_times else 0.0
            dep_dt = pd.to_datetime(grp.iloc[0]['departure_time'])
            dep_day_hour = int(dep_dt.dayofweek * 24 + dep_dt.hour)
            day_hour = []
            for j in sampled_idx:
                ts = pd.to_datetime(grp.iloc[j]['time'])
                dh = int(ts.dayofweek * 24 + ts.hour)
                day_hour.append(int(dh % 168))
        elif time_mode == 'scale_total_time':
            # Simple baseline policy:
            # TrajFlow exports evenly spaced per-point timestamps.
            # We scale the relative 0~119 minute axis to the real sparse trajectory duration.
            # arrival_times unit is days; t_start=0.0; t_end=last arrival_time.
            raw_seq = test_metadata[i]['raw_sequence']
            real_arrival_times = raw_seq.get('arrival_times', [])

            if len(real_arrival_times) >= 2:
                target_span_days = max(
                    float(real_arrival_times[-1]) - float(real_arrival_times[0]),
                    1e-8
                )
            else:
                target_span_days = 1.0 / 1440.0

            raw_rel = np.array(sampled_idx, dtype=float)
            raw_rel = raw_rel - raw_rel[0]

            if len(raw_rel) >= 2 and raw_rel[-1] > 0:
                arrival_times = (raw_rel / raw_rel[-1] * target_span_days).tolist()
            else:
                arrival_times = [0.0] * len(sampled_idx)

            t_start = 0.0
            t_end = arrival_times[-1] if arrival_times else 0.0

            dep_dt = pd.to_datetime(grp.iloc[0]['departure_time'])
            day_hour = []
            for at in arrival_times:
                ts = dep_dt + pd.Timedelta(days=float(at))
                dh = int(ts.dayofweek * 24 + ts.hour) % 168
                day_hour.append(dh)

        lat_out = [p[0] for p in sampled_gps]
        lon_out = [p[1] for p in sampled_gps]
        seq_idx = test_metadata[i].get('seq_idx', i)

        # Section 9 sanity check: checkins must exist in poi_gps (flexible key)
        for j, c in enumerate(checkins):
            key_found = (
                c in poi_gps
                or (isinstance(c, (int, str)) and int(c) in poi_gps)
                or (isinstance(c, (int, str)) and str(c) in poi_gps)
            )
            if not key_found:
                raise ValueError(
                    f"Trajectory {i} point {j}: checkin POI id={c!r} not found in poi_gps. "
                    f"POI id type={type(c)}. poi_gps keys sample: {list(poi_gps.keys())[:5]}"
                )

        # Section 9 sanity check: marks must be valid category values.
        # marks are category IDs (int), not POI IDs.
        # Validate against the set of unique category values observed in poi_category.
        for j, m in enumerate(marks):
            if m not in valid_categories:
                raise ValueError(
                    f"Trajectory {i} point {j}: mark category={m!r} not in valid category set. "
                    f"Valid categories sample: {list(valid_categories)[:10]}"
                )

        seq_dict = {
            'arrival_times': arrival_times,
            'marks': marks,
            'checkins': checkins,
            'gps': sampled_gps,
            'day_hour': day_hour,
            'lat': lat_out,
            'lon': lon_out,
            'revisit': revisit,
            'seq_idx': [seq_idx],
            't_start': t_start,
            't_end': t_end,
        }
        generated_sequences.append(seq_dict)

    # 7. GPS bounds check (P1-5: lenient, summarize only)
    print("\n[7/7] Computing GPS bounds check and distance statistics...")
    traj_gps_lists = [seq['gps'] for seq in generated_sequences]
    if skip_gps_check:
        print("  --skip_gps_check is set: skipping strict GPS bounds enforcement.")
    else:
        check_gps_bounds_batch(traj_gps_lists, lat_min, lat_max, lon_min, lon_max, padding_deg=0.05)

    # GPS->POI Haversine distance statistics
    dists = np.array(all_gps_distances)
    dist_min    = float(np.min(dists))
    dist_median = float(np.median(dists))
    dist_p90    = float(np.percentile(dists, 90))
    dist_max    = float(np.max(dists))

    print(f"\n  GPS->POI Haversine distance (km):")
    print(f"    min    = {dist_min:.4f}")
    print(f"    median = {dist_median:.4f}")
    print(f"    p90    = {dist_p90:.4f}")
    print(f"    max    = {dist_max:.4f}")

    if time_mode == 'csv_time':
        print(f"\n  INFO: arrival_times computed from TrajFlow CSV timestamps (csv_time mode).")
    elif time_mode == 'scale_total_time':
        print(
            f"\n  INFO: scale_total_time mode -- TrajFlow does not explicitly generate "
            f"irregular event timestamps; we linearly scale its exported relative timestamp "
            f"axis to the target sparse trajectory duration.")

    if dist_median > 1.0 or dist_p90 > 5.0:
        print(f"\n  WARNING: GPS->POI distance is large (median={dist_median:.3f}km, p90={dist_p90:.3f}km).")
        print(f"  Possible causes: denormalization error, wrong grid encoding, or model collapse.")
    else:
        print(f"\n  GPS->POI distances look reasonable.")

    # Write output
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump({'sequences': generated_sequences}, f)
    print(f"\n  Written: {output_pkl}")
    print(f"  Total sequences: {len(generated_sequences)}")

    print(f"\n{'='*60}")
    print(f"Postprocessing complete: {city}")
    print(f"  Output: {output_pkl}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Postprocess TrajFlow CSV -> MIRAGE generated.pkl")
    parser.add_argument('--city', type=str, required=True,
                       choices=['NewYork', 'Tokyo', 'Istanbul'])
    parser.add_argument('--csv', type=str, default=None,
                       help='Explicit path to generated_trajectories.csv')
    parser.add_argument('--time_mode', type=str, default='scale_total_time',
                       choices=['csv_time', 'scale_total_time'],
                       help='How to compute arrival_times. Default: scale_total_time. '
                            'scale_total_time linearly scales TrajFlow relative timestamps to '
                            'the target sparse trajectory duration.')
    parser.add_argument('--skip_gps_check', action='store_true', default=False,
                       help='Skip strict GPS bounds enforcement. Use only for debug / pipeline '
                            'verification when the model is undertrained (e.g. few epochs).')
    args = parser.parse_args()
    postprocess(args.city, args.csv, args.time_mode, args.skip_gps_check)


if __name__ == '__main__':
    main()
