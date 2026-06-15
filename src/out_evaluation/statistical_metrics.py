"""
统计指标评估模块 (Statistical Metrics Evaluation Module)

本模块实现轨迹生成质量的统计评估指标，包括：
1. 原有指标：Distance(移动距离)、Radius(活动半径)、DailyLoc(每日地点数)、
             Interval(时间间隔)、Category(类别分布)
2. 新增指标：
   - Hourly: 24小时活动分布 JSD（检测时间规律性）
   - SpatialKDE: 空间核密度估计 JSD（检测空间分布相似性）

权威来源：
- JSD (Jensen-Shannon Divergence): 信息论标准对称散度
- Hourly Distribution: Pappalardo et al., scikit-mobility
- Spatial KDE: 核密度估计是空间分析的标准方法 (Ripley's K-function 等)
"""

import numpy as np
from scipy import stats


def distance(lat1, lon1, lat2, lon2):
    """
    计算两个GPS坐标点之间的球面距离 (Haversine公式)
    
    Args:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度
    
    Returns:
        距离（单位：公里）
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # 地球平均半径（公里）
    return c * r

def travel_distance(geo):
    return np.sum(distance(geo[:-1, 0], geo[:-1, 1], geo[1:, 0], geo[1:, 1]))


def radius(geo):
    center = np.mean(geo, axis = 0)
    return np.sqrt(np.mean(distance(geo[:, 0], geo[:, 1], center[0], center[1])))


def JSD(P_A, P_B):
    epsilon = 1e-14
    P_A = (P_A / P_A.sum() + epsilon)
    P_B = (P_B / P_B.sum() + epsilon)
    P_merged = 0.5 * (P_A + P_B)
    
    kl_PA_PM = np.sum(P_A * np.log(P_A / P_merged))
    kl_PB_PM = np.sum(P_B * np.log(P_B / P_merged))
    
    jsd = 0.5 * (kl_PA_PM + kl_PB_PM)
    return jsd

def arr_to_distribution(arr, min, max, bins):
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution

def compute_probability_distribution(data):
    unique_elements, counts = np.unique(data, return_counts=True)
    total_counts = np.sum(counts)
    probabilities = counts / total_counts
    return unique_elements, probabilities

def category_jsd(generated_category, real_category):
    gen_category, prob_gen = compute_probability_distribution(generated_category)
    real_category, prob_real = compute_probability_distribution(real_category)

    p,q = (list(zip(gen_category, prob_gen)), list(zip(real_category, prob_real)))

    p = np.asarray(p)
    q = np.asarray(q)

    all_elements = set(p[:, 0]).union(set(q[:, 0]))
    p_probs = {element: 0.0 for element in all_elements}
    q_probs = {element: 0.0 for element in all_elements}
    
    for element, prob in p:
        p_probs[element] = prob
    
    for element, prob in q:
        q_probs[element] = prob

    jsd_value = JSD(np.array(list(p_probs.values())),np.array(list(q_probs.values())))
    return jsd_value


def evaluation(generated, original):
    generated = np.array(generated)
    original = np.array(original)
    max = np.max(generated) if np.max(generated) > np.max(original) else np.max(original)
    p_gen = arr_to_distribution(generated, 0, max, 100)
    p_real = arr_to_distribution(original, 0, max, 100)
    jsd = JSD(p_gen, p_real)
    return jsd

def get_visits(trajs,max_locs):
    visits = np.zeros(shape=(max_locs), dtype=float)
    for t in trajs:
        visits[t] += 1
    visits = visits / np.sum(visits)
    return visits

def get_topk_visits(visits, K):
    locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
    locs_visits.sort(reverse=True, key=lambda d: d[1])
    topk_locs = [locs_visits[i][0] for i in range(K)]
    topk_probs = [locs_visits[i][1] for i in range(K)]
    return np.array(topk_probs), topk_locs


# =============================================================================
# 新增指标 1: 24小时活动分布 JSD (Hourly Activity Distribution JSD)
# =============================================================================

def compute_hourly_distribution_jsd(real_data, generated_data):
    """
    计算真实轨迹与生成轨迹的24小时活动分布JSD
    
    核心思想：
    - 人类活动有明显的昼夜节律（circadian rhythm）
    - 真实轨迹通常呈现双峰分布（早高峰 + 晚高峰）
    - 生成轨迹若不捕捉时间规律，会呈现均匀或异常分布
    
    实现方法：
    1. 从 arrival_times（以天为单位）提取小时信息：hour = (arrival_time % 1) * 24
    2. 统计24个时间槽的活动频次
    3. 归一化为概率分布
    4. 计算 JSD
    
    权威来源：
    - Pappalardo et al., "scikit-mobility: A Python library for mobility analysis"
    - Song et al., "Limits of Predictability in Human Mobility", Science 2010
    
    Args:
        real_data: 真实轨迹列表，每条轨迹包含 'arrival_times' 和 'day_hour' 字段
        generated_data: 生成轨迹列表，格式同上
    
    Returns:
        JSD 值（越小表示分布越相似，0 表示完全相同）
    """
    def extract_hourly_distribution(data):
        """从轨迹数据中提取24小时活动分布"""
        hourly_counts = np.zeros(24)
        
        for seq in data:
            if 'day_hour' in seq and seq['day_hour'] is not None:
                day_hours = np.array(seq['day_hour'])
                day_hours = day_hours[day_hours >= 0]
                hours = np.floor(day_hours).astype(int) % 24
            else:
                arrival_times = np.array(seq['arrival_times'])
                hours = np.floor((arrival_times % 1) * 24).astype(int) % 24
            
            for hour in hours:
                hourly_counts[hour] += 1
        
        if hourly_counts.sum() > 0:
            hourly_probs = hourly_counts / hourly_counts.sum()
        else:
            hourly_probs = np.ones(24) / 24
        
        return hourly_probs
    
    real_hourly = extract_hourly_distribution(real_data)
    gen_hourly = extract_hourly_distribution(generated_data)
    
    return JSD(real_hourly, gen_hourly)


# =============================================================================
# 新增指标 2: 空间核密度估计 JSD (Spatial KDE JSD)
# =============================================================================

def compute_spatial_kde_jsd(real_data, generated_data, grid_size=50):
    """
    计算真实轨迹与生成轨迹的空间密度分布JSD
    
    核心思想：
    - 人类活动在空间上呈现不均匀分布（热点区域 vs 稀疏区域）
    - 使用核密度估计（KDE）将离散点转换为连续密度场
    - 在同一网格上比较两个密度分布的 JSD
    
    Args:
        real_data: 真实轨迹列表，每条轨迹包含 'gps' 字段
        generated_data: 生成轨迹列表，格式同上
        grid_size: 网格大小，默认50x50=2500个评估点
    
    Returns:
        JSD 值（越小表示分布越相似，0 表示完全相同）
    """
    def extract_all_gps_points(data):
        all_lats = []
        all_lons = []
        
        for seq in data:
            if 'gps' in seq and len(seq['gps']) > 0:
                gps = np.array(seq['gps'])
                all_lats.extend(gps[:, 0].tolist())
                all_lons.extend(gps[:, 1].tolist())
        
        return np.array(all_lats), np.array(all_lons)
    
    real_lats, real_lons = extract_all_gps_points(real_data)
    gen_lats, gen_lons = extract_all_gps_points(generated_data)
    
    if len(real_lats) < 2 or len(gen_lats) < 2:
        print("Warning: 数据点不足，无法计算空间KDE JSD")
        return 1.0
    
    lat_min = min(real_lats.min(), gen_lats.min())
    lat_max = max(real_lats.max(), gen_lats.max())
    lon_min = min(real_lons.min(), gen_lons.min())
    lon_max = max(real_lons.max(), gen_lons.max())
    
    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05
    lat_min -= lat_margin
    lat_max += lat_margin
    lon_min -= lon_margin
    lon_max += lon_margin
    
    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
    positions = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()])
    
    try:
        real_kde = stats.gaussian_kde(np.vstack([real_lats, real_lons]))
        gen_kde = stats.gaussian_kde(np.vstack([gen_lats, gen_lons]))
        
        real_density = real_kde(positions)
        gen_density = gen_kde(positions)
        
        real_probs = real_density / real_density.sum()
        gen_probs = gen_density / gen_density.sum()
        
        return JSD(real_probs, gen_probs)
        
    except np.linalg.LinAlgError:
        print("Warning: KDE 计算失败（数据可能退化），使用直方图方法")
        
        real_hist, _, _ = np.histogram2d(real_lats, real_lons, bins=grid_size, 
                                          range=[[lat_min, lat_max], [lon_min, lon_max]])
        gen_hist, _, _ = np.histogram2d(gen_lats, gen_lons, bins=grid_size,
                                         range=[[lat_min, lat_max], [lon_min, lon_max]])
        
        real_probs = real_hist.ravel() / real_hist.sum()
        gen_probs = gen_hist.ravel() / gen_hist.sum()
        
        return JSD(real_probs, gen_probs)


# =============================================================================
# 主评估函数（已扩展包含新指标）
# =============================================================================

def Get_Statistical_Metrics(real_data, generated_data, min_seq_len=1):
    """
    计算轨迹生成质量的全面统计评估指标
    
    包含7个评估维度：
    1. Distance:   移动距离分布 JSD（Haversine公式计算轨迹总长度）
    2. Radius:     活动半径分布 JSD（回转半径，衡量活动范围）
    3. DailyLoc:   每日访问地点数分布 JSD
    4. Interval:   时间间隔分布 JSD（事件间的时间差）
    5. Category:   POI类别分布 JSD
    6. Hourly:     24小时活动分布 JSD（新增，检测时间规律性）
    7. SpatialKDE: 空间核密度分布 JSD（新增，检测空间热点分布）
    
    Args:
        real_data: 真实轨迹列表
        generated_data: 生成轨迹列表
        min_seq_len: 最小序列长度阈值
    
    Returns:
        dict: 包含所有指标的JSD值字典
    """
    JSD_Results = {
        'Distance': 1.0,
        'Radius': 1.0,
        'DailyLoc': 1.0,
        'Interval': 1.0,
        'Category': 1.0,
        'Hourly': 1.0,
        'SpatialKDE': 1.0
    }

    assert len(generated_data) > 0, "生成数据不能为空"
    assert len(real_data) > 0, "真实数据不能为空"

    # 【共享口径】五个核心指标全部委托给 get_five_official_metrics 计算
    # 这保证了离线测试与训练期 val_gen 使用完全相同的代码路径
    five = get_five_official_metrics(real_data, generated_data, min_seq_len)
    JSD_Results['Distance'] = five['distance_jsd']
    JSD_Results['Radius']   = five['radius_jsd']
    JSD_Results['DailyLoc'] = five['dailyloc_jsd']
    JSD_Results['Interval']  = five['interval_jsd']
    JSD_Results['Category']  = five['category_jsd']

    JSD_Results['Hourly'] = compute_hourly_distribution_jsd(real_data, generated_data)
    JSD_Results['SpatialKDE'] = compute_spatial_kde_jsd(real_data, generated_data)
    
    # 诊断：回访率
    def compute_revisit_rate(data):
        total_events = 0
        total_revisits = 0
        for seq in data:
            if 'revisit' in seq and seq['revisit'] is not None:
                revisit = np.array(seq['revisit'])
                if len(revisit) > 1:
                    revisit_mask = revisit[1:]
                    total_events += len(revisit_mask)
                    total_revisits += np.sum(revisit_mask == 1)
        return total_revisits / total_events if total_events > 0 else 0.0
    
    real_revisit_rate = compute_revisit_rate(real_data)
    gen_revisit_rate = compute_revisit_rate(generated_data)
    
    print("\n" + "=" * 50)
    print("【诊断】回访率 (Revisit Rate)")
    print("=" * 50)
    print(f"真实数据回访率: {real_revisit_rate:.4f} ({real_revisit_rate*100:.2f}%)")
    print(f"生成数据回访率: {gen_revisit_rate:.4f} ({gen_revisit_rate*100:.2f}%)")
    print(f"差异: {abs(gen_revisit_rate - real_revisit_rate):.4f}")
    
    # 诊断：类别边缘分布
    def compute_category_distribution(data):
        from collections import Counter
        all_marks = []
        for seq in data:
            if 'marks' in seq and seq['marks'] is not None:
                all_marks.extend(list(seq['marks']))
        if len(all_marks) == 0:
            return {}
        counter = Counter(all_marks)
        total = len(all_marks)
        return {k: v/total for k, v in sorted(counter.items())}
    
    real_cat_dist = compute_category_distribution(real_data)
    gen_cat_dist = compute_category_distribution(generated_data)
    
    print("\n" + "=" * 50)
    print("【诊断】类别边缘分布 (Category Distribution)")
    print("=" * 50)
    all_cats = sorted(set(real_cat_dist.keys()) | set(gen_cat_dist.keys()))
    print(f"{'类别':>6} | {'真实':>8} | {'生成':>8} | {'差异':>8}")
    print("-" * 40)
    total_diff = 0
    for cat in all_cats:
        real_pct = real_cat_dist.get(cat, 0) * 100
        gen_pct = gen_cat_dist.get(cat, 0) * 100
        diff = abs(gen_pct - real_pct)
        total_diff += diff
        print(f"{cat:>6} | {real_pct:>7.2f}% | {gen_pct:>7.2f}% | {diff:>+7.2f}%")
    print("-" * 40)
    print(f"总偏差 (L1): {total_diff:.2f}%")
    
    return JSD_Results


# =============================================================================
# 共享函数：从 FlowDataset item 或生成结果构建标准 sequence dict
# 供 train_flow.py 的 generation-level validation 与最终离线测试共用同一套口径
# =============================================================================

def _build_sequence_from_flow_item(item, dataset):
    """
    将 FlowDataset 的单条归一化 item 还原为与最终测试格式一致的 dict。

    输入 item 字段（FlowDataset.__getitem__ 输出）：
        x1:             [MAX_SEQ_LEN, 3]  归一化轨迹（lat_norm, lon_norm, dt_log_norm）
        marks:           [MAX_SEQ_LEN]     category（-100 = padding）
        valid_length:    int
        cond_continuous: [9]             意图向量
        start_day_hour:  int
        revisit:         [MAX_SEQ_LEN]
        target_poi:     [MAX_SEQ_LEN]     dense vocab id
        raw_checkins:    [MAX_SEQ_LEN]     raw POI id
        poi_loss_mask:   [MAX_SEQ_LEN]

    返回 dict（与最终测试 Get_Statistical_Metrics 期望的格式一致）：
        arrival_times: list[float]   累积到达时间（天为单位）
        gps:           list[list]    [[lat, lon], ...]
        checkins:      list[int/str] raw POI id
        marks:         list[int]     category id
        day_hour:      list[int]     0~167
        t_start:       float         起始绝对时间（day）
        t_end:         float         结束绝对时间（day）
    """
    vl = int(item.get('valid_length', 0))
    if vl <= 0:
        return None

    x1 = item.get('x1')
    if hasattr(x1, 'cpu'):
        x1 = x1.cpu().numpy()
    else:
        x1 = np.asarray(x1)
    x1 = x1[:vl]

    lat_norm = x1[:, 0]
    lon_norm = x1[:, 1]
    dt_log_norm = x1[:, 2]

    lat = (lat_norm + 1.0) * (dataset.lat_max - dataset.lat_min) / 2.0 + dataset.lat_min
    lon = (lon_norm + 1.0) * (dataset.lon_max - dataset.lon_min) / 2.0 + dataset.lon_min
    dt_log = dt_log_norm * dataset.dt_std + dataset.dt_mean
    delta = np.exp(dt_log) - 1e-5
    delta = np.maximum(delta, 1e-5)
    arrival = np.cumsum(delta)

    start_day_hour = item.get('start_day_hour', 0)
    if isinstance(start_day_hour, (int, float)):
        sd = int(start_day_hour)
    elif hasattr(start_day_hour, 'item'):
        sd = int(start_day_hour.item())
    else:
        sd = int(start_day_hour)

    day_hour_arr = (sd + np.floor(arrival * 24)).astype(int) % 168

    gps = np.stack([lat, lon], axis=1).tolist()
    arrival_list = arrival.tolist()

    marks_raw = item.get('marks')
    if marks_raw is None:
        marks = [1] * vl
    else:
        if hasattr(marks_raw, 'cpu'):
            marks_raw = marks_raw.cpu().numpy()
        else:
            marks_raw = np.asarray(marks_raw).ravel()
        marks_raw = marks_raw[:vl]
        marks = [int(m) if int(m) != -100 else 1 for m in marks_raw]

    raw_checkins = item.get('raw_checkins')
    if raw_checkins is not None:
        if hasattr(raw_checkins, 'cpu'):
            raw_checkins = raw_checkins.cpu().numpy()
        else:
            raw_checkins = np.asarray(raw_checkins)
        checkins = [int(c) for c in raw_checkins[:vl]]
    else:
        checkins = [1] * vl

    return {
        'arrival_times': arrival_list,
        'gps': gps,
        'checkins': checkins,
        'marks': marks,
        'day_hour': day_hour_arr.tolist(),
        't_start': 0.0,
        't_end': float(arrival[-1]) if len(arrival) > 0 else 0.0,
    }


def _build_sequence_from_gen_output(
    traj_norm, start_day_hour, valid_length,
    lat_min, lat_max, lon_min, lon_max,
    dt_mean, dt_std,
    decoded_checkins=None,
    decoded_marks=None,
):
    """
    将 train_flow.py 中生成的单条轨迹还原为标准 sequence dict。

    参数：
        traj_norm: [vl, 3] 归一化轨迹（lat_norm, lon_norm, dt_log_norm）
        start_day_hour: int 起始 day_hour
        valid_length: int
        lat_min/lat_max/lon_min/lon_max/dt_mean/dt_std: 归一化参数
        decoded_checkins: list[int] 可选，解码出的 raw POI id 列表（用于生成数据）
        decoded_marks: list[int] 可选，POI→category 映射后的 category 列表

    返回 dict（与 _build_sequence_from_flow_item 输出格式一致）
    """
    vl = int(valid_length)
    if vl <= 0:
        return None

    lat_norm = traj_norm[:vl, 0]
    lon_norm = traj_norm[:vl, 1]
    dt_log_norm = traj_norm[:vl, 2]

    lat = (lat_norm + 1.0) * (lat_max - lat_min) / 2.0 + lat_min
    lon = (lon_norm + 1.0) * (lon_max - lon_min) / 2.0 + lon_min
    dt_log = dt_log_norm * dt_std + dt_mean
    delta = np.exp(dt_log) - 1e-5
    delta = np.maximum(delta, 1e-5)
    arrival = np.cumsum(delta)

    sd = int(start_day_hour) if not hasattr(start_day_hour, 'item') else int(start_day_hour.item())
    day_hour_arr = (sd + np.floor(arrival * 24)).astype(int) % 168

    gps = np.stack([lat, lon], axis=1).tolist()
    arrival_list = arrival.tolist()

    if decoded_checkins is not None:
        checkins = [int(c) for c in decoded_checkins[:vl]]
    else:
        checkins = [1] * vl

    if decoded_marks is not None:
        marks = [int(m) for m in decoded_marks[:vl]]
    else:
        marks = [1] * vl

    return {
        'arrival_times': arrival_list,
        'gps': gps,
        'checkins': checkins,
        'marks': marks,
        'day_hour': day_hour_arr.tolist(),
        't_start': 0.0,
        't_end': float(arrival[-1]) if len(arrival) > 0 else 0.0,
    }


def get_five_official_metrics(real_seqs, gen_seqs, min_seq_len=1):
    """
    【共享函数】计算五个官方 JSD 指标，口径与最终离线测试完全一致。

    该函数是 train_flow.py generation-level validation 与最终离线测试
    evaluation.py → run_Statistical → Get_Statistical_Metrics 的唯一共同调用路径。
    不允许在这条路径之外手写任何 proxy metric。

    参数：
        real_seqs: list[dict]  真实轨迹列表（格式见 _build_sequence_from_flow_item）
        gen_seqs:  list[dict]  生成轨迹列表（格式见 _build_sequence_from_gen_output）
        min_seq_len: int       最小序列长度阈值（默认 1）

    返回 dict：
        {
            'distance_jsd': float,
            'radius_jsd':  float,
            'interval_jsd': float,
            'dailyloc_jsd': float,
            'category_jsd': float,
            'mean_jsd': float,
        }
    """
    assert len(gen_seqs) > 0, "生成数据不能为空"
    assert len(real_seqs) > 0, "真实数据不能为空"

    Real_Stats = {'Distance': [], 'Radius': [], 'DailyLoc': [], 'Interval': [], 'Category': []}
    Gen_Stats = {'Distance': [], 'Radius': [], 'DailyLoc': [], 'Interval': [], 'Category': []}

    seqs_data = [gen_seqs, real_seqs]
    stats_dicts = [Gen_Stats, Real_Stats]

    for idx, seqs in enumerate(seqs_data):
        for seq in seqs:
            checkins = seq.get('checkins', [])
            if len(checkins) < min_seq_len:
                continue
            gps = np.array(seq['gps'])

            # Distance: travel_distance = sum of consecutive Haversine distances
            sd = Gen_Stats if idx == 0 else Real_Stats
            sd['Distance'].append(travel_distance(gps))
            sd['Radius'].append(radius(gps))
            sd['DailyLoc'].append(len(set(checkins)))

            # Interval: 全局 ediff1d 扁平化，与最终测试完全一致
            t_start = seq.get('t_start', 0.0)
            t_end = seq.get('t_end', seq['arrival_times'][-1] if seq['arrival_times'] else 0.0)
            arrival_times = seq.get('arrival_times', [])
            intervals = np.ediff1d(np.concatenate([
                np.array([t_start]),
                np.array(arrival_times),
                np.array([t_end])
            ]))
            sd['Interval'].extend(intervals.tolist())
            sd['Category'].extend(seq.get('marks', []))

    # 五个指标统一用最终测试口径
    results = {}
    for metric in ['Distance', 'Radius', 'DailyLoc', 'Interval', 'Category']:
        if metric == 'Category':
            results[metric.lower() + '_jsd'] = category_jsd(
                Gen_Stats[metric], Real_Stats[metric])
        else:
            results[metric.lower() + '_jsd'] = evaluation(
                Gen_Stats[metric], Real_Stats[metric])

    mean_jsd = (results['distance_jsd'] + results['radius_jsd'] +
                 results['interval_jsd'] + results['dailyloc_jsd'] +
                 results['category_jsd']) / 5.0
    results['mean_jsd'] = mean_jsd

    return results

