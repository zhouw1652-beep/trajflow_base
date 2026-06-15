"""
prepare_mirage_trajflow_data.py

Convert MIRAGE pkl format to TrajFlow standard processed format for the MIRAGE baseline.

Usage:
    python scripts/prepare_mirage_trajflow_data.py --city NewYork
    python scripts/prepare_mirage_trajflow_data.py --city Tokyo
    python scripts/prepare_mirage_trajflow_data.py --city Istanbul
    python scripts/prepare_mirage_trajflow_data.py --city NewYork --departure_encoding hour_z
"""

import argparse
import math
import os
import pickle
import json
import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
TRAJFLOW_ROOT = (SCRIPT_DIR / "..").resolve()
if str(TRAJFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAJFLOW_ROOT))

from scripts.mirage_io import load_mirage_pkl


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def geohash_encode(lat, lon, precision):
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    BITS = [16, 8, 4, 2, 1]
    result = []
    bit = 0
    ch = 0
    even = True
    while len(result) < precision:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                lon_range[0] = mid
                ch |= BITS[bit]
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                lat_range[0] = mid
                ch |= BITS[bit]
            else:
                lat_range[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            result.append(BASE32[ch])
            bit = 0
            ch = 0
    return ''.join(result)


def geohash_to_int(gh_str):
    BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    val = 0
    for c in gh_str:
        val = val * 32 + BASE32.index(c)
    return val


def parse_gps_coord(coord):
    if isinstance(coord, str):
        parts = coord.split(',')
        return float(parts[0]), float(parts[1])
    elif isinstance(coord, (list, tuple)):
        return float(coord[0]), float(coord[1])
    else:
        raise ValueError(f"Cannot parse GPS coord of type {type(coord)}: {coord}")


def resample_gps(gps_list, target_n):
    if not gps_list:
        return [[0.0, 0.0]] * target_n
    if len(gps_list) == target_n:
        return [list(p) for p in gps_list]
    lats = [p[0] for p in gps_list]
    lons = [p[1] for p in gps_list]
    old_idx = np.linspace(0, len(gps_list) - 1, len(gps_list))
    new_idx = np.linspace(0, len(gps_list) - 1, target_n)
    resampled_lats = np.interp(new_idx, old_idx, lats)
    resampled_lons = np.interp(new_idx, old_idx, lons)
    return [[float(resampled_lats[i]), float(resampled_lons[i])] for i in range(target_n)]


def uniform_sample(n_items, L):
    if L <= 0:
        return []
    if L >= n_items:
        return list(range(n_items))
    return [int(round(i)) for i in np.linspace(0, n_items - 1, L)]


def detect_departure_encoding(generate_py_path):
    if not os.path.exists(generate_py_path):
        raise RuntimeError(
            f"generate.py not found at {generate_py_path}. "
            "Run with --departure_encoding hour_z or --departure_encoding raw_bucket."
        )
    with open(generate_py_path, 'r', encoding='utf-8') as f:
        src = f.read()
    # Pattern 1: cond_info[i, 0] * 6 + 12 (literal)
    if 'cond_info[i, 0]' in src and '* 6 +' in src:
        return 'hour_z'
    # Pattern 2: cond_info[i, 0] * DEP_HOUR_STD + DEP_HOUR_MEAN (named constants)
    if ('cond_info[i, 0]' in src and 'DEP_HOUR_STD' in src and
            'DEP_HOUR_MEAN' in src and '* DEP_HOUR_STD +' in src):
        return 'hour_z'
    # Pattern 3: cond_info[i, 0] * 0.0833333 (raw_bucket)
    if 'cond_info[i, 0]' in src and '* 0.0833333' in src:
        return 'raw_bucket'
    if 'cond_info[i, 0]' in src and '/ 12' in src:
        return 'raw_bucket'
    raise RuntimeError(
        "Could not auto-detect departure encoding from generate.py.\n"
        "Expected 'cond_info[i, 0] * 6 + 12' (hour_z) or "
        "'cond_info[i, 0] * 0.0833333' (raw_bucket).\n"
        "Run with explicit --departure_encoding:\n"
        "  python scripts/prepare_mirage_trajflow_data.py --city {CITY} --departure_encoding hour_z"
    )


def build_geohash_grid(poi_gps, precision=6):
    """
    Returns (mesh_mapping_dict, grid_meta).
    mesh_mapping_dict: {geohash_int: compact_id}
      compact_id is 0..n_grids-1, assigned by sorted(geohash_int) order.
      Saved as {geohash_int: compact_id} because geohash_int is the stable
      canonical key; PrepareDataset.loadExistingData reverses it to
      {compact_id: geohash_int} at load time.
    """
    # Normalize keys to int (poi_gps may have int or str keys depending on how the pkl was saved)
    poi_gps_int = {}
    for pid, coord in poi_gps.items():
        k = int(pid)
        poi_gps_int[k] = coord
    poi_gps = poi_gps_int

    gh_seen = {}
    for pid, coord in poi_gps.items():
        lat, lon = parse_gps_coord(coord)
        gh = geohash_encode(lat, lon, precision)
        gh_int = geohash_to_int(gh)
        gh_seen.setdefault(gh_int, gh)
    unique_ghs = sorted(gh_seen.keys())
    mesh_mapping_dict = {gh_int: idx for idx, gh_int in enumerate(unique_ghs)}
    grid_meta = {"encoding": "geohash", "geohash_precision": precision}
    return mesh_mapping_dict, grid_meta


def split_indices(n, train_ratio=0.6, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    t = int(round(n * train_ratio))
    v = int(round(n * val_ratio))
    train_idx = sorted(perm[:t].tolist())
    val_idx = sorted(perm[t:t+v].tolist())
    test_idx = sorted(perm[t+v:].tolist())
    return train_idx, val_idx, test_idx


def process_sequences(sequences, poi_gps, poi_cat,
                    mesh_mapping_dict,
                    train_idx, val_idx,
                    departure_mode,
                    traj_length=120):
    n_seqs = len(sequences)
    traj_segments = [None] * n_seqs
    conditions_raw = [None] * n_seqs
    metadata = [None] * n_seqs
    missing_start = 0
    missing_end = 0

    train_total_dis = []
    train_total_time = []
    train_total_len = []
    train_avg_dis = []
    train_avg_speed = []
    train_set = set(train_idx)

    for idx, seq in enumerate(sequences):
        arrival_times = seq['arrival_times']
        gps = seq['gps']
        day_hours = seq['day_hour']
        original_len = len(gps)

        gps_resampled = resample_gps(gps, traj_length)
        traj_segments[idx] = np.array(gps_resampled, dtype=np.float32)

        # Departure (col 0)
        hour_float = float(day_hours[0] % 24)
        if departure_mode == 'hour_z':
            dep_raw = (hour_float - 12.0) / 6.0
        elif departure_mode == 'raw_bucket':
            dep_raw = float(int(hour_float * 12))
        else:
            dep_raw = 0.0

        # Condition cols 1-5: raw values
        if len(gps) < 2:
            total_dis_km = 0.0
        else:
            total_dis_km = sum(
                haversine_km(gps[i][0], gps[i][1], gps[i+1][0], gps[i+1][1])
                for i in range(len(gps) - 1)
            )
        total_dis_m = total_dis_km * 1000.0
        if len(arrival_times) >= 2:
            total_time_s = max((arrival_times[-1] - arrival_times[0]) * 86400.0, 1e-6)
        else:
            total_time_s = 1.0
        total_len = original_len
        avg_dis_m = total_dis_m / total_len if total_len > 0 else 0.0
        avg_speed_mps = total_dis_m / total_time_s if total_time_s > 0 else 0.0

        if idx in train_set:
            train_total_dis.append(total_dis_m)
            train_total_time.append(total_time_s)
            train_total_len.append(float(total_len))
            train_avg_dis.append(avg_dis_m)
            train_avg_speed.append(avg_speed_mps)

        # OD geohash (cols 6-7): compact id
        start_lat, start_lon = float(gps[0][0]), float(gps[0][1])
        end_lat, end_lon = float(gps[-1][0]), float(gps[-1][1])
        start_gh_int = geohash_to_int(geohash_encode(start_lat, start_lon, 6))
        end_gh_int = geohash_to_int(geohash_encode(end_lat, end_lon, 6))

        # P1-2: Track missing geohash mappings instead of silent fallback to 0
        if start_gh_int not in mesh_mapping_dict:
            missing_start += 1
            start_id = 0
        else:
            start_id = mesh_mapping_dict[start_gh_int]

        if end_gh_int not in mesh_mapping_dict:
            missing_end += 1
            end_id = 0
        else:
            end_id = mesh_mapping_dict[end_gh_int]

        conditions_raw[idx] = np.array([
            dep_raw,
            total_dis_m,
            total_time_s,
            float(total_len),
            avg_dis_m,
            avg_speed_mps,
            float(start_id),
            float(end_id),
            0.0,  # trans_mode = 0 (walk)
        ], dtype=np.float32)

        # day_hour from arrival_times fraction
        day_hour_arr = []
        if arrival_times:
            frac0 = arrival_times[0] % 1.0
            base_hour = int(frac0 * 24) % 24
            base_day = day_hours[0] - base_hour if day_hours else 0
            base_day = base_day - (base_day % 24)
            for at in arrival_times:
                h = int((at % 1.0) * 24)
                day_hour_arr.append(int((base_day + h) % 168))
        else:
            day_hour_arr = [0] * original_len

        seq_idx_val = seq.get('seq_idx', [idx])
        if isinstance(seq_idx_val, list):
            seq_idx_val = seq_idx_val[0]

        metadata[idx] = {
            'dataset_index': idx,
            'original_length': original_len,
            'seq_idx': seq_idx_val,
            'raw_sequence': seq,
            'departure_bucket': int(hour_float * 12),
            'hour_float': hour_float,
            'day_hour': day_hour_arr,
        }

    # cond_mean/std from TRAIN split only
    cond_mean = np.array([
        np.mean(train_total_dis),
        np.mean(train_total_time),
        np.mean(train_total_len),
        np.mean(train_avg_dis),
        np.mean(train_avg_speed),
    ], dtype=np.float32)
    cond_std = np.array([
        max(float(np.std(train_total_dis, ddof=1)), 1e-6),
        max(float(np.std(train_total_time, ddof=1)), 1e-6),
        max(float(np.std(train_total_len, ddof=1)), 1e-6),
        max(float(np.std(train_avg_dis, ddof=1)), 1e-6),
        max(float(np.std(train_avg_speed, ddof=1)), 1e-6),
    ], dtype=np.float32)

    # Z-score normalize cols 1-5 for ALL sequences
    conditions = []
    for i in range(n_seqs):
        c = conditions_raw[i].copy()
        for col_i, (m, s) in enumerate(zip(cond_mean, cond_std)):
            c[col_i + 1] = (c[col_i + 1] - m) / s
        conditions.append(c)

    # P1-2: Report geohash fallback statistics
    missing_start_ratio = missing_start / n_seqs if n_seqs > 0 else 0.0
    missing_end_ratio = missing_end / n_seqs if n_seqs > 0 else 0.0
    if missing_start > 0 or missing_end > 0:
        print(f"  WARNING: geohash fallback detected:")
        print(f"    missing_start={missing_start} ({missing_start_ratio:.2%}), "
              f"missing_end={missing_end} ({missing_end_ratio:.2%})")
    else:
        print(f"  Geohash mapping: all {n_seqs} trajectories mapped correctly (no fallback)")

    # P1-2: Raise if fallback rate exceeds 1%
    if missing_start_ratio > 0.01 or missing_end_ratio > 0.01:
        raise ValueError(
            f"Geohash fallback rate too high: "
            f"start={missing_start_ratio:.2%}, end={missing_end_ratio:.2%} (threshold: 1%). "
            f"Check that POI GPS coordinates cover all trajectory endpoints."
        )

    traj_segments_arr = np.array(traj_segments, dtype=np.float32)
    conditions_arr = np.array(conditions, dtype=np.float32)
    return traj_segments_arr, conditions_arr, cond_mean, cond_std, metadata


def write_outputs(output_dir, traj_segments, conditions,
                 cond_mean, cond_std,
                 grid_meta, mesh_mapping_dict,
                 train_idx, val_idx, test_idx,
                 metadata, departure_mode):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_lats = np.concatenate([t[:, 0] for t in traj_segments])
    all_lons = np.concatenate([t[:, 1] for t in traj_segments])
    traj_lat_mean = float(np.mean(all_lats))
    traj_lat_std = float(np.std(all_lats))
    traj_lon_mean = float(np.mean(all_lons))
    traj_lon_std = float(np.std(all_lons))

    with open(output_dir / 'traj_segments.pkl', 'wb') as f:
        pickle.dump(traj_segments, f)
    with open(output_dir / 'conditions.pkl', 'wb') as f:
        pickle.dump(conditions, f)
    with open(output_dir / 'mesh_mapping_dict.pkl', 'wb') as f:
        pickle.dump(mesh_mapping_dict, f)
    with open(output_dir / 'grid_meta.json', 'w') as f:
        json.dump(grid_meta, f, indent=2)

    with open(output_dir / 'traj_mean_std.txt', 'w') as f:
        f.write(f"lat_mean: {traj_lat_mean}\nlat_std: {traj_lat_std}\n"
                 f"lon_mean: {traj_lon_mean}\nlon_std: {traj_lon_std}\n")

    with open(output_dir / 'conditions_mean_std.txt', 'w') as f:
        names = ['total_dis', 'total_time', 'total_len', 'avg_dis', 'avg_speed']
        for name, m, s in zip(names, cond_mean, cond_std):
            f.write(f"{name}_mean: {m}\n{name}_std: {s}\n")

    with open(output_dir / 'split_train.pkl', 'wb') as f:
        pickle.dump(train_idx, f)
    with open(output_dir / 'split_val.pkl', 'wb') as f:
        pickle.dump(val_idx, f)
    with open(output_dir / 'split_test.pkl', 'wb') as f:
        pickle.dump(test_idx, f)

    train_val_idx = sorted(train_idx + val_idx)
    with open(output_dir / 'train_val_indices.pkl', 'wb') as f:
        pickle.dump(train_val_idx, f)
    with open(output_dir / 'test_indices.pkl', 'wb') as f:
        pickle.dump(test_idx, f)

    test_meta = [metadata[i] for i in test_idx]
    with open(output_dir / 'test_metadata.pkl', 'wb') as f:
        pickle.dump(test_meta, f)

    # Sanity checks
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(traj_segments)
    max_cid = int(conditions[:, 6].max())
    n_grids = len(mesh_mapping_dict)
    assert 0 <= max_cid < n_grids, f"Compact ID {max_cid} out of [0, {n_grids})"
    nan_count = int(np.isnan(traj_segments).sum())
    inf_count = int(np.isinf(traj_segments).sum())
    nan_cond = int(np.isnan(conditions).sum())
    assert nan_count == 0 and inf_count == 0 and nan_cond == 0, \
        f"NaN/Inf: traj={nan_count}/{inf_count}, cond={nan_cond}"

    print(f"\n=== Sanity Checks ===")
    print(f"  Shapes: traj_segments={traj_segments.shape}, conditions={conditions.shape}")
    print(f"  NaN/Inf: traj={nan_count}/{inf_count}, cond={nan_cond}")
    print(f"  Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Grid: compact_ids [0, {n_grids}), max_used={max_cid}")
    print(f"  cond_mean (from train): {cond_mean}")
    print(f"  cond_std  (from train): {cond_std}")
    print(f"  Departure mode: {departure_mode}")
    print(f"  NOTE: generate.py uses cond[0]*6+12 for hour_z; raw bucket in metadata.")
    print(f"  WARNING: trans_mode=0 (walk) -- MIRAGE has no transport mode.")
    print(f"  WARNING: POI/category/revisit projected via KDTree in postprocess.")
    print(f"  mesh_mapping_dict: saved as {{geohash_int: compact_id}}; "
          f"PrepareDataset.loadExistingData reverses to {{compact_id: geohash_int}}.")
    print(f"  cond[6/7] = compact_id; denormalization uses cr_sample_grid_mapping_dict "
          f"(same direction, compact_id -> geohash_int).")
    print(f"  cond_mean/std computed from TRAIN split only.")
    print(f"\n  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert MIRAGE pkl to TrajFlow format.")
    parser.add_argument('--city', type=str, required=True,
                       choices=['NewYork', 'Tokyo', 'Istanbul'])
    parser.add_argument('--departure_encoding', type=str, default='auto',
                       choices=['auto', 'hour_z', 'raw_bucket'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    city = args.city

    print(f"\n{'='*60}")
    print(f"MIRAGE TrajFlow Data Preparation - {city}")
    print(f"{'='*60}\n")

    # 1. Departure encoding
    if args.departure_encoding == 'auto':
        gen_py = TRAJFLOW_ROOT / 'generate.py'
        departure_mode = detect_departure_encoding(str(gen_py))
        print(f"[auto] Detected departure encoding: {departure_mode}")
    else:
        departure_mode = args.departure_encoding
        print(f"[explicit] Using: {departure_mode}")

    # 2. Load
    print(f"\n[1/5] Loading MIRAGE pkl for {city}...")
    mirage_obj = load_mirage_pkl(city, TRAJFLOW_ROOT)
    sequences = mirage_obj['sequences']
    poi_gps = mirage_obj['poi_gps']
    poi_cat = mirage_obj['poi_category']
    print(f"  {len(sequences)} sequences, {len(poi_gps)} POIs, {len(poi_cat)} categories")

    # 3. Geohash grid
    print(f"\n[2/5] Building geohash-6 grid...")
    mesh_mapping_dict, grid_meta = build_geohash_grid(poi_gps, precision=6)
    print(f"  {len(mesh_mapping_dict)} unique geohash cells")
    print(f"  mesh_mapping_dict format: {{geohash_int: compact_id}}")

    # 4. Split FIRST (needed for train-only cond_mean/std)
    print(f"\n[3/5] Splitting (seed={args.seed})...")
    train_idx, val_idx, test_idx = split_indices(len(sequences), seed=args.seed)
    print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 5. Process sequences
    print(f"\n[4/5] Processing sequences...")
    traj_segments, conditions, cond_mean, cond_std, metadata = process_sequences(
        sequences, poi_gps, poi_cat, mesh_mapping_dict,
        train_idx, val_idx, departure_mode)
    print(f"  traj_segments={traj_segments.shape}, conditions={conditions.shape}")

    # 6. Write
    print(f"\n[5/5] Writing output files...")
    output_dir = TRAJFLOW_ROOT / 'data' / 'mirage_trajflow' / city
    write_outputs(output_dir, traj_segments, conditions,
                  cond_mean, cond_std, grid_meta, mesh_mapping_dict,
                  train_idx, val_idx, test_idx, metadata, departure_mode)

    print(f"\n{'='*60}")
    print(f"Preparation complete for {city}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
