import argparse
import os
import glob
from dpp.evaluations.preprocessing import preprocessing_data
from dpp.evaluations import Get_Statistical_Metrics,run_SemLoc_task,run_EpiSim_task
import torch
import pandas as pd
import numpy as np

# 【v6 patch】用 __file__ 定位脚本所在目录，避免 os.getcwd() 受终端工作目录影响
script_dir = os.path.dirname(os.path.abspath(__file__))
mirage_dir = os.path.dirname(script_dir)   # MIRAGE/code/ → MIRAGE/
dataset_file_path = mirage_dir


# =====================================================================
# PyTorch 版本兼容工具函数
# =====================================================================
def safe_torch_load(filepath, encoding=None, **kwargs):
    """
    兼容 PyTorch 1.x ~ 2.x 的 torch.load 封装。
    始终使用 weights_only=False 以兼容服务器端高版本 PyTorch（2.6+），
    同时在本地 PyTorch 1.x 下不传入该参数以避免报错。
    """
    import torch
    load_kwargs = {}
    if encoding is not None:
        load_kwargs['encoding'] = encoding
    try:
        load_kwargs['weights_only'] = False
        return torch.load(filepath, **load_kwargs, **kwargs)
    except TypeError:
        load_kwargs.pop('weights_only', None)
        return torch.load(filepath, **load_kwargs, **kwargs)


def _strip_ckpt_prefix(experiment_comments):
    """剥离 best_ / final_ 前缀，还原 base_name"""
    base_name = experiment_comments
    for prefix in ["best_", "final_"]:
        if experiment_comments.startswith(prefix):
            base_name = experiment_comments[len(prefix):]
            break
    return base_name


def _get_ckpt_mode(experiment_comments):
    """从 experiment_comments 中提取 ckpt_mode（best / final）"""
    for prefix in ["best_", "final_"]:
        if experiment_comments.startswith(prefix):
            return prefix.rstrip('_')
    return None


def _find_gen_file(dataset, experiment_comments):
    """
    自动查找生成文件。

    生成文件的实际命名规则（来自 generate_flow.py）：
      {data_name}_{experiment_comments}_generated.pkl

    例如 experiment_comments="best_robust_v1_istanbul" → 生成文件名中的 ckpt_mode 为 "best"
    例如 experiment_comments="final_robust_v1_istanbul" → 生成文件名中的 ckpt_mode 为 "final"

    因此必须根据 experiment_comments 中的前缀来推断正确的 ckpt_mode，
    不能硬编码遍历顺序（否则 best 和 final 会混用同一文件）。
    """
    # 从 experiment_comments 提取 ckpt_mode（best / final）
    ckpt_mode = _get_ckpt_mode(experiment_comments)
    base_name = _strip_ckpt_prefix(experiment_comments)

    if ckpt_mode:
        # 生成文件命名: {data_name}_{experiment_comments}_generated.pkl
        # 例如: Istanbul_best_robust_v1_istanbul_generated.pkl
        candidate = f'{dataset_file_path}/data/{dataset}/{dataset}_{experiment_comments}_generated.pkl'
        if os.path.exists(candidate):
            return candidate

    # Fallback: 直接用 base_name（兼容无前缀的情况）
    fallback = f'{dataset_file_path}/data/{dataset}/{dataset}_{base_name}_generated.pkl'
    return fallback


def collect_results(dataset_list, task_lists):
    def scan_result_files(file_dir):
        list_files = []
        for files in os.listdir(file_dir):
            list_files.append(files)
        return list_files
    results = []
    for dataset_name in dataset_list:
        data_frame = pd.DataFrame(columns=['method', 'mrr@5', 'mrr@10', 'ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'])
        for model in task_lists:
            files = scan_result_files(f"./log/{model}")
            file = [f for f in files if dataset_name in f][0]
            with open(f'./log/{model}/{file}', 'r') as f:
                data = f.readlines()
                res_list = eval((data[-1].strip('\n').split(': ')[1])[12:-1])
                data = [item[1] for item in res_list]
                data.insert(0, model)
                data_frame.loc[len(data_frame)] = data
        results.append(data_frame)
    real_res = results[0].iloc[:, 1:]
    gene_res = results[1].iloc[:, 1:]
    absolute_differences = (gene_res - real_res).abs()
    relative_differences = absolute_differences / real_res
    MAPE = relative_differences.mean().mean()
    relative_differences_2 = np.square(relative_differences)
    MSPE = relative_differences_2.mean().mean()
    return MAPE, MSPE


def run_LocRec(task, dataset, cuda, experiment_comments, split_comments, generated_only):
    print(f'preprocessing data for {task} Task')
    base_name = _strip_ckpt_prefix(split_comments)
    ckpt_mode = _get_ckpt_mode(experiment_comments)
    test_data_name = f'{dataset}_{base_name}_test'
    if ckpt_mode:
        gen_data_name = f'{dataset}_{experiment_comments}_generated'
    else:
        gen_data_name = f'{dataset}_{base_name}_generated'
    datasets = [f'{test_data_name}.pkl', f'{gen_data_name}.pkl']
    datasets_prefix = [f'{test_data_name}_for_general', f'{gen_data_name}_for_general']
    for source_file in datasets:
        preprocessing_data(f'{dataset_file_path}/data/{dataset}', source_file, False)
    LocRec_Tasks = ['DMF', 'LightGCN', 'MultiVAE', 'NeuMF', 'BPR']
    LocRec_settings = ['0', '0', '0', '0', '0']
    for index, dataset_name in enumerate(datasets_prefix):
        if generated_only and index == 0:
            continue
        for idx, model_name in enumerate(LocRec_Tasks):
            cmd = (f'python ./dpp/evaluations/run_LocRec.py --savepath {dataset_file_path}/data/{dataset} --model_name {model_name} --dataset_name {dataset_name}  --change_setting {LocRec_settings[idx]} --cuda {cuda}')
            os.system(cmd)
    LocRec_MAPE, LocRec_MSPE = collect_results(datasets_prefix, LocRec_Tasks)
    return LocRec_MAPE, LocRec_MSPE


def run_NexLoc(task, dataset, cuda, experiment_comments, split_comments, generated_only):
    print(f'preprocessing data for {task} Task')
    base_name = _strip_ckpt_prefix(split_comments)
    ckpt_mode = _get_ckpt_mode(experiment_comments)
    test_data_name = f'{dataset}_{base_name}_test'
    if ckpt_mode:
        gen_data_name = f'{dataset}_{experiment_comments}_generated'
    else:
        gen_data_name = f'{dataset}_{base_name}_generated'
    datasets = [f'{test_data_name}.pkl', f'{gen_data_name}.pkl']
    datasets_prefix = [f'{test_data_name}_for_sequential', f'{gen_data_name}_for_sequential']
    for source_file in datasets:
        preprocessing_data(f'{dataset_file_path}/data/{dataset}', source_file, True)
    NexLoc_Tasks = ['FPMC', 'BERT4Rec', 'Caser', 'SRGNN', 'SASRec']
    NexLoc_settings = ['0', '1', '1', '1', '1']
    for index, dataset_name in enumerate(datasets_prefix):
        if generated_only and index == 0:
            continue
        for idx, model_name in enumerate(NexLoc_Tasks):
            cmd = (f'python ./dpp/evaluations/run_NexLoc.py --savepath {dataset_file_path}/data/{dataset} --model_name {model_name} --dataset_name {dataset_name}  --change_setting {NexLoc_settings[idx]} --cuda {cuda}')
            os.system(cmd)
    NexLoc_MAPE, NexLoc_MSPE = collect_results(datasets_prefix, NexLoc_Tasks)
    return NexLoc_MAPE, NexLoc_MSPE


def run_SemLoc(task, dataset, experiment_comments, split_comments):
    print(f'run {task} Task')
    base_name = _strip_ckpt_prefix(split_comments)
    test_data_path = f'{dataset_file_path}/data/{dataset}/{dataset}_{base_name}_test.pkl'
    gen_data_path = _find_gen_file(dataset, experiment_comments)
    test_data = safe_torch_load(test_data_path, encoding='latin1')
    generated_data = safe_torch_load(gen_data_path, encoding='latin1')
    test_seqs = test_data.get('sequences')
    generated_seqs = generated_data.get('sequences')
    data_all = safe_torch_load(f'{dataset_file_path}/data/{dataset}/{dataset}.pkl', encoding='latin1')
    poi_label_dict = data_all['poi_category']
    SemLoc_MAPE, SemLoc_MSPE = run_SemLoc_task(test_seqs, generated_seqs, poi_label_dict)
    return SemLoc_MAPE, SemLoc_MSPE


def run_EpiSim(task, dataset, experiment_comments, split_comments, init_exposed_num, exp_num, max_weeks):
    print(f'run {task} Task')
    if 'Tokyo' in dataset:
        max_weeks = 12
    base_name = _strip_ckpt_prefix(split_comments)
    test_data_path = f'{dataset_file_path}/data/{dataset}/{dataset}_{base_name}_test.pkl'
    gen_data_path = _find_gen_file(dataset, experiment_comments)
    test_data = safe_torch_load(test_data_path, encoding='latin1')
    generated_data = safe_torch_load(gen_data_path, encoding='latin1')
    test_seqs = test_data.get('sequences')
    generated_seqs = generated_data.get('sequences')
    EpiSim_MAPE, EpiSim_MSPE = run_EpiSim_task(test_seqs, generated_seqs, init_exposed_num, exp_num, max_weeks)
    return EpiSim_MAPE, EpiSim_MSPE


def run_Statistical(dataset, experiment_comments, split_comments):
    print(f'get Statistical Performance')
    base_name = _strip_ckpt_prefix(split_comments)
    test_data_path = f'{dataset_file_path}/data/{dataset}/{dataset}_{base_name}_test.pkl'
    gen_data_path = _find_gen_file(dataset, experiment_comments)
    test_data = safe_torch_load(test_data_path, encoding='latin1')
    generated_data = safe_torch_load(gen_data_path, encoding='latin1')
    test_seqs = test_data.get('sequences')
    generated_seqs = generated_data.get('sequences')
    JSD_Values = Get_Statistical_Metrics(test_seqs, generated_seqs)
    return JSD_Values


def evaluation(task, dataset, cuda, results_log, experiment_comments, split_comments, generated_only, init_exposed_num, exp_num, max_weeks):
    if task != "Stat":
        if task == "LocRec":
            MAPE, MSPE = run_LocRec(task, dataset, cuda, experiment_comments, split_comments, generated_only)
        elif task == "NexLoc":
            MAPE, MSPE = run_NexLoc(task, dataset, cuda, experiment_comments, split_comments, generated_only)
        elif task == "SemLoc":
            MAPE, MSPE = run_SemLoc(task, dataset, experiment_comments, split_comments)
        elif task == "EpiSim":
            MAPE, MSPE = run_EpiSim(task, dataset, experiment_comments, split_comments, init_exposed_num, exp_num, max_weeks)
        with open(results_log, "a+") as f:
            f.writelines(f"{dataset} {task} MAPE : {MAPE}, MSPE : {MSPE} \n")
    else:
        JSD_Values = run_Statistical(dataset, experiment_comments, split_comments)
        with open(results_log, "a+") as f:
            f.writelines(
                f"Distance: {JSD_Values['Distance']}, "
                f"Radius: {JSD_Values['Radius']}, "
                f"Interval: {JSD_Values['Interval']}, "
                f"DailyLoc: {JSD_Values['DailyLoc']}, "
                f"Category: {JSD_Values['Category']}, "
                f"Hourly: {JSD_Values['Hourly']}, "
                f"SpatialKDE: {JSD_Values['SpatialKDE']}\n"
            )
        print("=" * 70)
        print("Statistical Metrics (JSD, lower is better):")
        print("-" * 70)
        print(f"  Distance:    {JSD_Values['Distance']}")
        print(f"  Radius:      {JSD_Values['Radius']}")
        print(f"  Interval:    {JSD_Values['Interval']}")
        print(f"  DailyLoc:    {JSD_Values['DailyLoc']}")
        print(f"  Category:    {JSD_Values['Category']}")
        print(f"  Hourly:      {JSD_Values['Hourly']}  (24h活动分布)")
        print(f"  SpatialKDE:  {JSD_Values['SpatialKDE']}  (空间密度分布)")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='Stat', type=str, choices=['LocRec', 'NexLoc', 'SemLoc', 'EpiSim', 'Stat'])
    parser.add_argument('--cuda', default=0, type=str)
    parser.add_argument('--datasets', default='NewYork', type=str)
    parser.add_argument('--results', default='', type=str)
    parser.add_argument("--experiment_comments", type=str, default="")
    parser.add_argument("--split_comments", type=str, default="",
        help="Experiment name for split-file lookup (default=experiment_comments). "
             "Use this when --experiment_comments includes suffixes that don't exist in split filenames.")
    parser.add_argument('--init_exposed_num', default=50, type=int)
    parser.add_argument('--exp_num', default=15, type=int)
    parser.add_argument('--max_weeks', default=15, type=int)
    parser.add_argument("--generated_only", default=False, action='store_true')
    args = parser.parse_args()
    if len(args.results) == 0:
        args.results = args.datasets + "_" + args.experiment_comments + "_Evaluation_results.txt" if args.experiment_comments != '' else args.datasets + "_Evaluation_results.txt"
    split_comments = args.split_comments if args.split_comments else args.experiment_comments
    evaluation(args.task, args.datasets, args.cuda, args.results, args.experiment_comments, split_comments, args.generated_only, args.init_exposed_num, args.exp_num, args.max_weeks)
