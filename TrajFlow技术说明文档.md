# TrajFlow 技术说明文档：基于 Flow Matching 的 GPS 轨迹生成框架

> **文档定位**：代码级技术说明，事实来源为仓库当前源码。代码与 README/论文描述不一致时，以代码为准。
>
> **阅读前提**：本文档面向具备深度学习基础的工程师，需要了解生成模型和轨迹建模的基本概念。
>
> **范围**：本文档仅描述 TrajFlow 框架本身的核心能力。POI/category/revisit 等语义字段不在 TrajFlow 原生输出范围内，需要在上层后处理。

---

## 1. 项目目标与任务定义

TrajFlow 是一个 **GPS 轨迹生成框架**，基于 Flow Matching（条件最优传输路径）。

- **输入**：processed 轨迹数据（固定长度轨迹点序列）+ 9 维条件向量
- **输出**：生成的 GPS 轨迹，最终保存为 CSV
- **不原生支持**：POI/category/revisit 等语义字段；这些需要在上层后处理模块中单独处理

### 核心 pipeline

```
原始 GPS 数据 → [预处理] → processed files
                                ↓
训练 pipeline ──────────────────→ FlowMatchingDataset
                                ↓
                        ConditionalVelocityModel
                                ↓
                    FlowMatchingTrainer → best_model.pt
                                ↓
推理 pipeline ──────────────────→ 生成轨迹 CSV
```

---

## 2. 仓库整体结构

```
TrajFlow/
├── train.py                    # 训练入口
├── generate.py                 # 生成入口（含 departure time 编码逻辑）
│
├── src/
│   ├── config/                 # 配置文件（config.yaml 等）
│   ├── data/
│   │   ├── dataset.py          # FlowMatchingDataset
│   │   └── transforms.py      # point2para / para2point
│   ├── models/
│   │   ├── networks.py        # 所有模型定义
│   │   └── wrappers.py         # 模型包装器
│   ├── training/
│   │   ├── trainer.py         # FlowMatchingTrainer
│   │   └── loss.py            # 损失函数（占位符，trainer 实际内联计算）
│   ├── eval/
│   │   └── inference.py       # FlowMatchingInference
│   └── out_evaluation/
│       ├── evaluation.py       # 框架级评测（参考 DPP 项目）
│       └── statistical_metrics.py  # 统计指标参考实现
│
├── data/                       # toy data（全合成假数据）
└── data_utils/
    ├── PrepareDataset.py      # 数据加载
    └── MiniTools.py           # 工具函数
```

---

## 3. 数据格式与 processed files

所有 processed files 位于 `data/<dataset_folder>/`，必须包含：

```
traj_segments.pkl
conditions.pkl
mesh_mapping_dict.pkl
traj_mean_std.txt
conditions_mean_std.txt
grid_meta.json         ← 可选，但强烈建议提供
```

### 3.1 traj_segments.pkl

**文件格式**：`pickle`，`numpy.ndarray`，dtype float32

**Shape**：`[N, trajectory_length, 2]`（`N` = 轨迹数量，axis 2 = `[latitude, longitude]`）

`traj_segments.pkl` 存的是什么值**完全取决于预处理方式**：

- **raw GPS**：存原始经纬度
- **已标准化**：存经过 `standardize_trajectories_preserve_aspect` 处理后的值
- **toy data**：经过 `standardize_trajectories_preserve_aspect` 处理

框架对坐标的标准化方式没有强制性约束，只要最终传入 `FlowMatchingDataset` 的数据格式正确（shape 正确、NaN/Inf 检查通过）即可。

### 3.2 conditions.pkl

**文件格式**：`pickle`，`numpy.ndarray`，dtype float32

**Shape**：`[N, 9]`

**9 维 condition 各列定义**：

| 列索引 | 字段名 | 含义（约定） | 是否标准化 |
|:------:|--------|------|:----------:|
| 0 | `departure` | 出发时间——**编码方式由预处理决定** | **不确定**（见 3.2.1） |
| 1 | `total_dis` | 总距离（约定为米，具体由预处理决定） | Z-score 标准化 |
| 2 | `total_time` | 总时长（约定为秒，具体由预处理决定） | Z-score 标准化 |
| 3 | `total_len` | 轨迹点数 | Z-score 标准化 |
| 4 | `avg_dis` | 平均点间距（约定为米，具体由预处理决定） | Z-score 标准化 |
| 5 | `avg_speed` | 平均速度（约定为米/秒，具体由预处理决定） | Z-score 标准化 |
| 6 | `starting_location` | 起点区域 ID（整数） | **无**（见 mesh_mapping_dict 说明） |
| 7 | `ending_location` | 终点区域 ID（整数） | **无**（见 mesh_mapping_dict 说明） |
| 8 | `trans_mode` | 交通方式（0=WALK, 1=CAR, 2=BUS, 3=TRAIN, 4=BIKE） | **无** |

#### 3.2.1 condition[0] departure 的编码方式

**预处理决定编码方式**，框架本身不对此做强制约束。当前代码中存在两种编码：

| 编码方式 | 公式 | 用于 |
|----------|------|------|
| `hour_z`（默认） | `(hour_float - 12.0) / 6.0` | `generate.py` save_trajectories_to_csv 的默认反推逻辑 |
| `raw_bucket` | `int(hour_float * 12)`（bucket ID ∈ [0, 287]） | toy data |

**generate.py 的反推逻辑**（`generate.py` save_trajectories_to_csv 函数）：

```python
DEP_HOUR_MEAN = 12.0
DEP_HOUR_STD  = 6.0
dep_z = float(cond_info[i, 0])
dep_hour_raw = dep_z * DEP_HOUR_STD + DEP_HOUR_MEAN
dep_hour = max(0, min(23, int(round(dep_hour_raw))))
```

`generate.py` **假设 condition[0] 存的是 `hour_z` 格式**。如果预处理使用了 `raw_bucket`，反推会给出错误的小时值。

**判定规则**：

```python
if 0 <= conditions[0, 0] <= 287 and conditions[0, 0] == int(conditions[0, 0]):
    # 存的是 raw_bucket 格式 → generate.py 反推会出错
elif -3 <= conditions[0, 0] <= 3:
    # 存的是 hour_z 格式 → generate.py 反推正确
```

#### 3.2.2 已知冲突

- `generate.py` 使用 `hour_z` 反推（`cond * 6 + 12`）
- `data/make_toy_data.py` 使用 `raw_bucket` 存储

使用 toy data 训练并用 generate.py 生成的 CSV 中，`departure_time` 会是错误的值。

### 3.3 mesh_mapping_dict.pkl

#### 3.3.1 数据结构与 key/value 不变量

`mesh_mapping_dict.pkl` 的格式：**`{key: value}` 对的 key/value 角色取决于预处理脚本和框架代码之间的约定**，不能独立理解。

**pipeline 中涉及 mesh_mapping_dict 的变量汇总**：

| 变量名 | 类型 | 在 pipeline 中的角色 | key | value |
|--------|------|---------------------|-----|--------|
| `mesh_mapping_dict.pkl`（源文件） | `dict[int→int]` | 预处理写入 | `raw_cell_id`（原始 geohash int 或 jismesh int） | `compact_id`（从 0 开始的序号） |
| `dataset.grid_mapping_dict` | `dict[int→int]` | `PrepareDataset.loadExistingData` 加载后，key/value **互换** | `compact_id`（源文件 value） | `raw_cell_id`（源文件 key） |
| `dataset.conditions[:, 6:8]` | `ndarray` | 存 **compact OD ID**（从 0 开始的连续整数） | — | — |
| `dataset.onehot_mapping_dict` | `dict[int→int]` | `map_two_columns_to_shared_range` 输出 | 原始 OD compact ID | 新的从 0 开始的重映射 ID |
| `dataset.cr_sample_grid_mapping_dict` | `dict[int→int]` | inference 反查：`{重映射ID: 原始geohash/jismesh int}` | 重映射 ID | 原始 geohash/jismesh int |
| `inference.py` 反查 | — | 用 `cr_sample_grid_mapping_dict` 反推 | 重映射 ID | 原始 geohash/jismesh int |

**PrepareDataset 的 key/value 互换**（`PrepareDataset.py` loadExistingData）：

```python
grid_mapping_dict = MiniTools.loadPKL(GRID_MAPPING_PATH)
grid_mapping_dict = {v: k for k, v in grid_mapping_dict.items()}
# 互换后: {compact_id: raw_cell_id}
```

**dataset._prepare_conditions 的 OD 重映射**：

```python
# conditions[:, 6:8] 原本存 grid_mapping_dict 的 key（原始 OD 编码）
# map_two_columns_to_shared_range 将其重映射为从 0 开始的连续整数
self.conditions[:, 6:8], self.onehot_mapping_dict, max_unique_length = \
    map_two_columns_to_shared_range(self.conditions[:, 6:8])

# 构建 cr_sample_grid_mapping_dict: {新ID: 原始geohash_int}
for key, value in onehot_mapping_dict.items():
    cr_sample_grid_mapping_dict[value] = self.grid_mapping_dict[key]
```

**inference 反查时的完整路径**：

```
condition[i, 6] = 重映射ID (0, 1, 2, ...)
    ↓
cr_sample_grid_mapping_dict[重映射ID] = 原始geohash_int
    ↓
用原始geohash_int + jismesh.to_meshpoint 或 geohash解码 → 经纬度
```

#### 3.3.2 迁移新数据集时的关键要求

- `mesh_mapping_dict.pkl` 的 key/value 角色必须与上述约定一致
- `PrepareDataset.loadExistingData` 会对 key/value 做互换
- `dataset._prepare_conditions` 会再次对 OD ID 做 `map_two_columns_to_shared_range`
- `inference.denormalize_trajectories` 用 `cr_sample_grid_mapping_dict` 做反向查询

**任何跳过 `PrepareDataset` 直接加载 pkl 的代码**，都必须自行维护这套 key/value 不变量。

### 3.4 traj_mean_std.txt 和 conditions_mean_std.txt

**traj_mean_std.txt**：经纬度标准化参数。是否被使用取决于 `norm1by1` 配置——当 `norm1by1=True` 时不使用（见 3.5）。

**conditions_mean_std.txt**：仅覆盖 conditions 列 1-5（total_dis, total_time, total_len, avg_dis, avg_speed）。列 0（departure）和列 6-8（OD, trans_mode）不在其中。

### 3.5 norm1by1 的标准化/反标准化机制

当 `data.norm1by1=True` 时：

**训练时（dataset.py `standardize_trajectories_preserve_aspect`）**：

```python
# 以质心为原点，整体标准差缩放（保持宽高比）
std = sqrt(mean(||p - mean||^2))  # 标量
standardized_traj = (traj - mean) / std
```

**不是**全局 Z-score，**不使用** `traj_mean_std.txt`。

**推理时反标准化（inference.py `denormalize_trajectories`）**：

走 MixStrategy 路径，基于 OD 端点坐标做相似/仿射变换还原，也不使用 `traj_mean_std.txt`。

**当 `norm1by1=False` 时**：数据使用全局 Z-score（`traj_mean_std.txt`），但反标准化分支可能不完整（`generate.py` 第 291-293 行的简单乘 std 加 mean 仅在 `norm1by1=False` 且非条件模式下触发）。

---

## 4. Config 系统

### 4.1 配置块总览

| 配置块 | 说明 |
|--------|------|
| `project` | name, output_dir, seed |
| `data` | 数据路径与处理方式（见 4.2） |
| `flow_matching` | Flow Matching 开关与超参数（与 ddpm/baseline 互斥） |
| `ddpm` | DDPM 开关与超参数（与 flow_matching/baseline 互斥） |
| `condition` | 条件编码开关与方式（见 4.3） |
| `model` | 模型类型（unet/mlp/cnn/transformer/bilstm）与维度 |
| `training` | 训练超参数 |
| `inference` | 推理步数与方法 |
| `unet` | U-Net 超参数 |
| `visualization` | 可视化开关 |
| `baseline` | Baseline 模型开关（与 flow_matching/ddpm 互斥） |

### 4.2 data 块关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_folder` | str | `""` | 优先于 region |
| `trajectory_length` | int | `120` | 每条轨迹的点数 |
| `batch_size` | int | `256` | 训练 batch size |
| `sample_count` | int | `-1` | -1 = 使用全部数据 |
| `parametrized` | bool | `true` | 是否参数化（RDP-K 压缩） |
| `parametrized_method` | str | `rdp_k` | 参数化方法 |
| `parametrized_M` | int | `10` | 参数化后的关键点数 |
| `norm1by1` | bool | `true` | 逐条质心居中+缩放标准化 |
| `od_finer` | bool | `false` | 是否编码 OD 网格内偏移（4 维） |
| `geohash` | bool | `false` | 影响 WideAndDeep 中 OD 条件的编码分支（见 6.3 节）；具体输入格式应以 `networks.py::WideAndDeep.forward` 的实际实现为准 |
| `geohash_precision` | int | `6` | geohash 精度 |

**model input_dim 计算**：

```python
M = parametrized_M if parametrized else trajectory_length
input_dim = M * 2           # 基本维度
if od_finer: input_dim += 4  # 额外 OD finer 参数
```

### 4.3 condition 块关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用条件输入 |
| `condition_type` | str | `"full"` | `"full"`=全部9维；`"od"`=仅OD 2维 |
| `transportation_mode` | bool | `true` | 是否包含 trans_mode embedding |
| `embedding_type` | str | `"wide_and_deep"` | 编码器类型 |
| `cfg_scale` | float | `1` | Classifier-free guidance scale |

### 4.4 互斥关系

`flow_matching.enabled`、`ddpm.enabled`、`baseline.enabled` **三者互斥**，且必须恰好启用一个。代码在 `train.py` 中强制检查：

```python
if len(enabled_models) != 1:
    raise ValueError("...")
```

---

## 5. Dataset 加载与预处理链路

### 5.1 FlowMatchingDataset 初始化流程

```python
# dataset.py FlowMatchingDataset.__init__
def __init__(self, config, mode='train'):
    # mode 参数在当前代码中未使用
    # train 和 eval 模式走完全相同的路径

    self.traj_length = config['data']['trajectory_length']
    self.dataset_size = config['data']['sample_count']

    # 加载轨迹数据
    self._load_trajectory_data(config)

    # 准备条件（OD 重映射等）
    if self.conditional:
        self._prepare_conditions()

    # 参数化（RDP-K 压缩）
    if config['data']['parametrized']:
        # 优先加载缓存的 .npy 文件
        # 否则调用 _convert_to_coefficients
```

### 5.2 _load_trajectory_data 详解

```python
# dataset.py _load_trajectory_data
def _load_trajectory_data(self, config):
    # 1. 确定数据文件夹
    # 2. 调用 PrepareDataset.loadExistingData
    (self.all_head,          # conditions.pkl [N, 9]
     self.traj_mean,          # [lat_mean, lon_mean]（来自 traj_mean_std.txt）
     self.traj_std,           # [lat_std, lon_std]
     self.lengths,            # 各轨迹原始长度（已反标准化）
     self.cond_mean,         # [5个均值]（conditions 列 1-5）
     self.cond_std,          # [5个标准差]
     self.traj_segments,      # [N, 120, 2] 轨迹数组
     self.grid_mapping_dict  # {compactID: geohash_int}
    ) = PrepareDataset.loadExistingData(self.input_folder, resample_length)

    # 3. 加载 grid_meta.json（确定 encoding: geohash vs jismesh）
    # 4. 限制数据集大小（sample_count 截断）
    # 5. od_finer 参数计算（如启用）
    # 6. 标准化（如 norm1by1=True）
    if config['data']['norm1by1']:
        self.standardize_trajectories_preserve_aspect()
```

### 5.3 norm1by1 标准化

```python
# dataset.py standardize_trajectories_preserve_aspect
for traj in self.traj_segments:
    mean = traj.mean(axis=0)          # (2,)
    centered = traj - mean
    std = sqrt(mean(||p||^2))          # 标量
    self.traj_segments[i] = centered / std
```

所有轨迹均值=0，标准差≈1，原点=质心。

### 5.4 _prepare_conditions

```python
# dataset.py _prepare_conditions
# condition_type='full': 全部9维
# condition_type='od': 仅OD 2维

# 步骤1: conditions[:, 6:8] → map_two_columns_to_shared_range
# 将原始 OD compact ID 重映射为从 0 开始的连续整数
self.conditions[:, 6:8], self.onehot_mapping_dict, max_unique_length = \
    map_two_columns_to_shared_range(self.conditions[:, 6:8])

# 步骤2: 构建 cr_sample_grid_mapping_dict
# {新ID: 原始geohash_int}
for key, value in onehot_mapping_dict.items():
    cr_sample_grid_mapping_dict[value] = self.grid_mapping_dict[key]
self.cr_sample_grid_mapping_dict = cr_sample_grid_mapping_dict

# 步骤3: 确定 embedding 维度
self.location_dim = max_unique_length
```

### 5.5 __getitem__ 输出

```python
# dataset.py __getitem__
# 主流配置（parametrized=True, od_finer=False, conditional=True）
# 返回: (traj_segment [M, 2], condition [9])
# DataLoader batch: x_1 [B, M, 2], condition [B, 9]

# od_finer=True 时
# 返回: (combined [M*2+4], condition [9])

# 无条件模式
# 返回: traj_segment [M, 2]
```

---

## 6. 模型结构

### 6.1 模型总览

| 类名 | 类型 | 说明 |
|------|------|------|
| `MLP` | 无条件 | 6 层全连接 |
| `CNN` | 无条件 | 1D CNN + time embedding |
| `TransformerVelocity` | 无条件 | Transformer Encoder |
| `BiLSTMVelocity` | 无条件 | BiLSTM |
| `TrajUnet` | 无条件 | U-Net（支持 DDPM/Flow Matching） |
| `ConditionalVelocityModel` | 条件 | 主流模型（封装 base + 条件编码器） |
| `WideAndDeep` | 条件编码器 | 9 维 condition → embedding |

**默认配置**：`ConditionalVelocityModel(type=unet) + WideAndDeep`

### 6.2 ConditionalVelocityModel forward（model_type='unet'）

```python
# networks.py ConditionalVelocityModel.forward
# x: [B, M*2] flatten
x = x.reshape(B, M, 2)        # [B, M, 2]
x = x.swapaxes(1, 2)           # [B, 2, M] ← U-Net conv1d 格式
c = condition_embedding(c)       # [B, embedding_dim]
y_t = unet(x, t, c)            # [B, 2, M]
y_t = y_t.swapaxes(1, 2)        # [B, M, 2]
return y_t  # velocity: [B, M, 2]
```

### 6.3 WideAndDeep 条件编码器

**`data.geohash` 配置影响 OD 编码方式**：

```python
# networks.py WideAndDeep.__init__
if config.data.geohash == True:
    # OD 使用 one-hot vector 编码，input_dim = location_dim
    self.sid_embedding = nn.Linear(location_emb_dim, hidden_dim)
    self.eid_embedding = nn.Linear(location_emb_dim, hidden_dim)
else:
    # OD 使用 Embedding 编码
    self.sid_embedding = nn.Embedding(location_emb_dim, hidden_dim)
    self.eid_embedding = nn.Embedding(location_emb_dim, hidden_dim)
```

**forward**：

```python
# networks.py WideAndDeep.forward(attr: [B, 9])
# 列 1-5 → Linear(5, embedding_dim)
wide_out = wide_fc(attr[:, 1:6])

# 列 0 → Embedding(288, hidden_dim)     departure
# 列 6 → Embedding/Linear（取决于 geohash）  starting_location
# 列 7 → Embedding/Linear（取决于 geohash）  ending_location
# 列 8 → Embedding(5, hidden_dim)      trans_mode（如启用）

# deep = concat(4 embeddings) → MLP → embedding_dim
# combined = wide_out + deep
return combined_embed  # [B, embedding_dim]
```

**encoding_bias = 0**：OD embedding 索引直接使用 compact ID，**不加偏移**。

---

## 7. Flow Matching 训练目标

### 7.1 数学框架

Linear Flow Matching（Conditional OT）：

$$
x_t = (1 - t) x_0 + t x_1, \quad t \sim U[0, 1], \quad x_0 \sim \mathcal{N}(0, I)
$$

$$
v_{\text{target}} = \frac{dx_t}{dt} = x_1 - x_0
$$

$$
\mathcal{L} = \mathbb{E} \left[ \| v_\theta(x_t, t, c) - v_{\text{target}} \|^2 \right]
$$

### 7.2 训练代码

```python
# trainer.py _train_step_flow_matching
x_1, condition = batch               # x_1: [B, M, 2], condition: [B, 9]
x_0 = torch.randn_like(x_1)          # [B, M, 2] ~ N(0,I)
t = torch.rand(B)                     # [B] ∈ [0,1]

path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
# path_sample.x_t  = (1-t)*x_0 + t*x_1
# path_sample.dx_t = x_1 - x_0

# Classifier-free guidance dropout（dropout_prob=0.1）
if dropout > 0 and rand() < dropout_prob:
    condition = zeros_like(condition)
pred_v = model(path_sample.x_t, path_sample.t, condition)

loss = (pred_v - path_sample.dx_t).pow(2).mean()
loss.backward(); optimizer.step()
```

**path 类型**：使用 `flow_matching` 库的 `AffineProbPath` + `CondOTScheduler`。

---

## 8. 训练入口 train.py

```python
def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))

    # 互斥检查
    # 创建 run_dir（时间戳格式）
    # 保存 config.yaml

    device = torch.device(...)

    # 创建 FlowMatchingDataset
    dataset = FlowMatchingDataset(config, mode='train')
    # model.input_dim 计算依赖 dataset 的 location_dim

    # 创建 ConditionalVelocityModel（必须传入 dataset）
    model = ConditionalVelocityModel(
        input_dim=dataset_params['trajectory_length']*2,
        hidden_dim=model_params['hidden_dim'],
        condition_dim=dataset.location_dim,  # 来自 dataset
        dataset=dataset,                     # 必须
        config=config
    )

    trainer = FlowMatchingTrainer(config, model, dataset, save_dir, device)
    trainer.train()
```

---

## 9. 推理与生成 generate.py

### 9.1 CLI 参数

```bash
python generate.py \
    --config ./outputs/run_YYYYMMDD_HHMMSS/config.yaml \
    --checkpoint ./outputs/models/run_.../best_model.pt \
    --generate_num 1000 \
    --batch_size 32 \
    --steps 10 \
    --method em
```

### 9.2 生成流程

```python
# generate.py main
def main():
    # 1. 加载 config 和 checkpoint
    # 2. 创建 FlowMatchingDataset（mode='eval'，与 train 路径相同）
    dataset = FlowMatchingDataset(config, mode='eval')

    # 3. 创建模型 + 加载权重
    model = ConditionalVelocityModel(..., dataset=dataset, ...)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 4. generate_trajectories
    for batch_data in dataloader:
        x_1, condition = batch_data

        # ODE 采样（Euler-Maruyama，实际 method_map 将 euler/rk4 都映射为 em）
        sol = inference.sample(condition=condition)

        # para2point 还原（如启用）
        # denormalize_trajectories（如启用）
        # save_trajectories_to_csv
```

### 9.3 save_trajectories_to_csv 详解

```python
# generate.py save_trajectories_to_csv
# CSV 字段: uid, departure_time, total_dis, total_time, total_len,
#          avg_dis, avg_speed, time, latitude, longitude

# departure 反推（假设 hour_z 格式）
DEP_HOUR_MEAN = 12.0; DEP_HOUR_STD = 6.0
dep_z = float(cond_info[i, 0])
dep_hour_raw = dep_z * DEP_HOUR_STD + DEP_HOUR_MEAN
dep_hour = clamp(int(round(dep_hour_raw)), 0, 23)
departure_time = f"2024-04-01 {dep_hour:02d}:00:00"

# time 字段: departure_time + j 分钟
time = pd.to_datetime(departure_time) + pd.Timedelta(minutes=j)
# 注意: j 从 0 到 trajectory_length-1，每点间隔固定 1 分钟
# 关键事实：此处的逐点时间没有用到 total_time / avg_speed / total_dis
# 也就是说，CSV 里的 time 列不是模型生成的，也不是按 total_time 展开的，
# 是 save_trajectories_to_csv 写文件时硬编码的占位规则。

# latitude/longitude: denormalize 后的坐标
# 如果 norm1by1=True 且 denorm_visualization=True，走 inference.denormalize_trajectories
# 否则简单乘 std 加 mean
```

---

## 10. 反标准化详解

### 10.1 端到端流程

```
conditions[:, 6:8] compact ID
    ↓
cr_sample_grid_mapping_dict[compact ID] → 原始 geohash/jismesh int
    ↓
jismesh.to_meshpoint 或 geohash解码 → OD 经纬度 [lat, lon]
    ↓
MixStrategy: 相似/仿射变换还原整条轨迹坐标
    ↓
输出: [B, 120, 2] 真实经纬度
```

### 10.2 MixStrategy 默认方法

```python
# inference.py denormalize_trajectories, DENORM_METHOD = 'MixStrategy'
# 自适应选择：
# - 各向异性严重（轴向缩放比 > 10）→ HybridSimilarity（等比缩放+旋转+端点校正）
# - 否则 → Affine_ExplicitParams（独立缩放 x/y 轴）
```

### 10.3 sampling_method 实际为 Euler-Maruyama

`generate.py` 的 `method_map` 将所有 CLI 方法映射为内部 `"em"`：

```python
method_map = {"euler": "em", "em": "em", "rk4": "em"}
# 即 euler 和 rk4 都走 Euler-Maruyama，rk4 名的 alias 是误导
```

---

## 11. Baseline / DDPM / VAE / GAN / Markov 支持

baseline（VAE/GAN/Markov）和 DDPM 与 Flow Matching **互斥**，需要分别启用。baseline 在 `generate.py` 中有独立加载和推理分支。**这些模型的完整性和稳定性未经过充分测试，真实场景推荐使用 Flow Matching**。

---

## 12. 端到端数据流图

### 12.1 训练 Pipeline

```
processed files
    ↓
PrepareDataset.loadExistingData()
    ↓
optional: standardize_trajectories_preserve_aspect()  [N, 120, 2]
    ↓
optional: RDP-K parameterization              [N, 10, 2]
    ↓
map_two_columns_to_shared_range (OD compact ID)
    ↓
FlowMatchingDataset
    ↓
DataLoader → batch: x_1 [B, M, 2], condition [B, 9]
    ↓
x_0 = randn([B, M, 2])
t = rand(B)
path_sample = path.sample(t, x_0, x_1)
# x_t = (1-t)*x_0 + t*x_1
# dx_t = x_1 - x_0
pred_v = model(x_t, t, condition)
loss = (pred_v - dx_t).pow(2).mean()
    ↓
backward + optimizer.step()
    ↓
best_model.pt
```

### 12.2 生成 Pipeline

```
checkpoint + config + processed files
    ↓
FlowMatchingDataset (mode='eval')
    ↓
ConditionalVelocityModel (with weights)
    ↓
sampling: x_init = randn([B, M, 2])
    ↓
Euler-Maruyama ODE integration (n_steps 步)
    ↓
para2point: [B, M, 2] → [B, 120, 2]
    ↓
denormalize_trajectories:
  OD ID → cr_sample_grid_mapping_dict → geohash/jismesh int → lat/lon
  MixStrategy: 相似/仿射变换
    ↓
save_trajectories_to_csv:
  departure: hour_z 反推 → 字符串时间
  time: departure + j*1min
  latitude/longitude: denormalize 后的坐标
    ↓
generated_trajectories.csv
```

---

## 13. 张量 Shape 速查

| 变量 | Shape | 说明 |
|------|-------|------|
| `traj_segments`（raw pkl） | `[N, 120, 2]` | 预处理后的坐标 |
| `traj_segments`（after parametrized） | `[N, 10, 2]` | RDP-K 压缩后 |
| `conditions` | `[N, 9]` | 9 维条件向量 |
| `batch.x_1` | `[B, 10, 2]` | 训练 batch |
| `batch.condition` | `[B, 9]` | 条件 batch |
| `x_0` | `[B, 10, 2]` | 标准正态噪声 |
| `x_t` | `[B, 10, 2]` | 插值中间状态 |
| `pred_v` | `[B, 10, 2]` | 预测速度 |
| `v_target` | `[B, 10, 2]` | x_1 - x_0 |

---

## 14. 迁移到新数据集的最小改动清单

### 14.1 数据文件准备

必须生成的文件：

1. `traj_segments.pkl`：`[N, trajectory_length, 2]`，dtype float32
2. `conditions.pkl`：`[N, 9]`，dtype float32
3. `mesh_mapping_dict.pkl`：OD 编码映射，key/value 角色见 3.3 节
4. `traj_mean_std.txt` 或确保 `norm1by1=True` 覆盖标准化逻辑
5. `conditions_mean_std.txt`：conditions 列 1-5 的 Z-score 参数
6. `grid_meta.json`：encoding 字段（"geohash" 或 "jismesh"）

### 14.2 关键检查项

- **departure 编码**：确认 `conditions[:, 0]` 与 `generate.py` 的反推逻辑一致（推荐 `hour_z`）
- **OD ID 映射**：确认 `mesh_mapping_dict.pkl` 的 key/value 角色与 `PrepareDataset.loadExistingData` 的互换逻辑一致
- **`standardize_trajectories_preserve_aspect`**：如果 pkl 存的是 raw GPS，需要在 Dataset 中启用 `norm1by1=True`
- **`data.geohash` 与网络编码方式一致**：确保 config 中 `data.geohash` 与预处理输出的 OD 编码方式匹配

### 14.3 Sanity Check

```python
import pickle, numpy as np

traj = pickle.load(open('data/your_data/traj_segments.pkl', 'rb'))
cond = pickle.load(open('data/your_data/conditions.pkl', 'rb'))

assert len(traj.shape) == 3 and traj.shape[2] == 2
assert cond.shape[1] == 9
assert 0 <= cond[:, 0].min() and cond[:, 0].max() <= 287 or \
       -3 <= cond[:, 0].min() and cond[:, 0].max() <= 3  # 两种格式之一
assert 0 <= cond[:, 8].min() and cond[:, 8].max() <= 4
```

---

## 15. 常见坑和排错指南

### 坑 1：departure 反推结果错误

**现象**：CSV 中 `departure_time` 全是凌晨或深夜的小时值。

**原因**：`generate.py` 假设 `conditions[:, 0]` 是 `hour_z` 格式（`(hour-12)/6`），但数据可能用了 `raw_bucket` 格式（整数 0-287）。

**检查**：打印 `conditions[0, 0]` 的值，如果大约是整数（0-287），说明存的是 raw_bucket。

**修复**：确保预处理时使用 `hour_z` 编码，或修改 `generate.py` 的反推逻辑。

---

### 坑 2：生成的坐标仍是 [-1,1] 范围

**现象**：CSV 中 latitude/longitude 值在 -1 到 1 之间。

**原因**：反标准化未执行或失败。

**检查**：`config['visualization']['norm1by12origialvis']` 是否为 True；`inference.denormalize_trajectories` 是否被调用。

---

### 坑 3：OD 坐标反查全错

**现象**：生成轨迹落在海洋或与预期完全不同的位置。

**原因**：`mesh_mapping_dict.pkl` 的 key/value 与 inference.py 的反查逻辑不匹配。

**检查**：打印 `cr_sample_grid_mapping_dict` 确认映射方向正确（key=重映射ID, value=原始geohash_int）。

---

### 坑 4：FlowMatchingDataset 的 mode 参数不区分数据集

**现象**：`mode='train'` 和 `mode='eval'` 行为相同。

**原因**：`mode` 参数在当前代码中未使用，train 和 eval 走完全相同的路径。框架本体不内置 train/test split 语义。

**说明**：如果需要 train/eval 自动加载不同数据子集，需要自己在 Dataset 或 DataLoader sampler 层实现，`mode` 参数本身不提供这一功能。

---

### 坑 5：euler / rk4 实际都走 Euler-Maruyama

**现象**：使用 `--method rk4` 期盼高精度积分但实际没有。

**原因**：`generate.py` 的 `method_map` 将 `euler` 和 `rk4` 都映射为 `em`。

---

### 坑 6：parametrized_M 与模型 input_dim 不一致

**现象**：`RuntimeError: size mismatch` 或 NaN loss。

**检查**：确认 `config['data']['parametrized_M']` 在 train.py 和 inference 中的使用一致。

---

### 坑 7：checkpoint 与 config 不匹配

**现象**：加载后输出全零或 NaN。

**检查**：checkpoint 中保存了 `config`，确保推理时使用的 config 与训练时一致。

---

### 坑 8：norm1by1=True 但 od_finer=False

**现象**：反标准化时 OD 坐标使用网格中心（mult=0.5），而非精确起终点。

**检查**：如果需要精确 OD 起终点，启用 `od_finer=True`（需要 jismesh 支持）。

---

### 坑 9：data.geohash 与网络编码方式不匹配

**现象**：训练 loss 正常但生成坐标全错，或 shape 不匹配。

**原因**：`config['data']['geohash']` 与预处理输出的 OD 编码方式不一致。`networks.py` 中 `WideAndDeep` 根据 `geohash=True/False` 选择 linear（one-hot）或 embedding 编码，如果配置与数据不匹配会导致维度错误或行为不符预期。

**检查**：确认 config 中 `data.geohash` 与 prepare 脚本输出的编码方式一致。

---

## 16. 限制与开放问题

1. **真实数据未开源**：BW 和 DiDi 数据不可用，toy data 是假数据
2. **departure 编码冲突**：`generate.py` 假设 `hour_z`，但 toy data 用 `raw_bucket`
3. **POI/category/revisit 不在 TrajFlow 原生输出范围内**：这些语义字段需要在上层后处理模块中单独建模
4. **评测脚本是参考实现**：`src/out_evaluation/` 中的 statistical_metrics.py 是参考实现，需要用户根据任务适配
5. **baseline/DDPM 模型完整度未充分验证**：推荐使用 Flow Matching
6. **逐点时间非模型生成**：`generate.py::save_trajectories_to_csv` 写 CSV 时，`time` 列硬编码为 `departure_time + j minutes`（`j ∈ [0, L-1]`），**完全未使用 `total_time` / `avg_speed` / `total_dis` 分配点间隔**。模型生成的状态维度是 `[M, 2]`，无显式时间维。论文中关于"按 provided travel time 做 length-consistent reconstruction"的说法，在开源 `generate.py` 的 CSV 输出阶段并未兑现。引用 TrajFlow 作为时间相关 baseline 时应明确：时间字段是 condition + 后处理规则构造的，不是模型逐点采样的。

---

## 17. 附录

### 17.1 训练/生成命令

```bash
# Toy smoke test
python data/make_toy_data.py
python train.py --config ./src/config/config_toy.yaml

# 生成
python generate.py \
    --config ./outputs/run_YYYYMMDD_HHMMSS/config.yaml \
    --checkpoint ./outputs/models/run_.../best_model.pt \
    --generate_num 1000 \
    --batch_size 32 \
    --steps 10
```

---

*文档完成时间：2026-06-13，基于 TrajFlow 仓库当前源码。*
