#!/usr/bin/env python3
"""
hyperparameter_search.py — Sync-X 超参数搜索
================================================================
基于 config.py 中定义的超参数搜索配置，执行顺序网格搜索。

策略（2026-08-11 修订，对齐论文契约）：
  论文锁定 λ_corr=0.3 / λ_shared=0.1 / lr=1e-4 / epochs 40/80 等（Ch3:97 + Ch4:169-173），
  不做主搜索；真正值得搜的是论文留白项：
  1. 先搜 λ_align × λ_uniform（论文 Ch4:171-172 "selected on val split"）—— loss_weights 阶段
  2. lr 阶段（通常只作确认，默认已是论文值 1e-4）
  3. training_epochs 阶段（⚠️ 运行本阶段时勿传 --epochs，否则搜索值 {40,80,120} 被覆盖空转）
  4. 最优配置 × ≥3 seeds 平均（论文 Ch4:250 要求 mean±std）

用法：
    # 快速筛选 λ_align × λ_uniform（9 组合，20 轮，~15min/实验 ≈ 2.3h）
    python hyperparameter_search.py --stage loss_weights --epochs 20

    # 固定最佳 λ，搜索 lr（确认用）
    python hyperparameter_search.py --stage learning_rate --epochs 40 \
        --best_lambda_align 0.1 --best_lambda_uniform 0.5

    # 固定最佳，搜训练轮数（勿传 --epochs！）
    python hyperparameter_search.py --stage training_epochs \
        --best_lambda_align 0.1 --best_lambda_uniform 0.5 --best_lr 1e-4

    # 最终：最优配置 × 3 seeds（80 轮，~3h）
    python hyperparameter_search.py --stage loss_weights --seeds "0 1 2" \
        --best_lambda_align 0.1 --best_lambda_uniform 0.5

输出：
    logs/hyperparameter_search/<stage>/ 目录下每个 (实验,seed) 的日志 + 独立 ckpt
    logs/hyperparameter_search/<stage>/summary.csv 汇总（含 val_auc 列，自动选优）
"""
import os
import sys
import argparse
import subprocess
import csv
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, HYPERPARAMETER_SEARCH_CONFIG, SEARCH_STRATEGY, DATASET_PATHS
from logging_utils import read_jsonl


def get_search_configs(stage_name):
    """获取指定阶段的搜索配置"""
    step = None
    for s in SEARCH_STRATEGY['steps']:
        if s['name'] == stage_name:
            step = s
            break
    if step is None:
        raise ValueError(f"Unknown stage: {stage_name}. Available: {[s['name'] for s in SEARCH_STRATEGY['steps']]}")
    
    param_names = step['params']
    param_configs = {name: HYPERPARAMETER_SEARCH_CONFIG[name] for name in param_names}
    
    # 生成所有组合
    values_lists = [pc['values'] for pc in param_configs.values()]
    combinations = list(product(*values_lists))
    
    configs = []
    for combo in combinations:
        cfg = Config()
        # 应用固定参数
        for key, val in step.get('fixed', {}).items():
            setattr(cfg, key, val)
        # 应用搜索参数
        for name, val in zip(param_names, combo):
            setattr(cfg, name, val)
        configs.append({
            'params': dict(zip(param_names, combo)),
            'config': cfg,
        })
    
    return configs, step


def read_best_metrics(output_dir, exp_id):
    """读 stage2_train.jsonl 最后一条含 val_auc 的记录（自动选优依据）。

    train_stage2.py:244-247 每 epoch 末落 val_auc/val_ap/val_eer/val_acc/val_f1。
    空 val / 未落盘 → 返回 None（调用方判 FAILED，不再静默记 SUCCESS）。
    """
    log = os.path.join(output_dir, "logs", exp_id, "stage2_train.jsonl")
    if not os.path.exists(log):
        return None
    try:
        recs = read_jsonl(log)
    except Exception:
        return None
    for r in reversed(recs):
        if r.get("phase") == "val" and "val_auc" in r:
            return {
                "val_auc": r.get("val_auc"),
                "val_ap": r.get("val_ap"),
                "val_eer": r.get("val_eer"),
            }
    return None


def run_single_experiment(cfg, output_dir, exp_id, epochs=None, stage1_ckpt="./ckpt/stage1.pt",
                          seed=0, batch=None):
    """运行单次实验（调用 train_stage2.py）。返回 {'success','val_auc','val_ap','val_eer'}。"""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, 'train_stage2.py',
        # DeepSpeak v1.1 无独立 val 划分，验证集复用 prep/test（见 prep_deepspeak.py）
        '--train_cache', f"{DATASET_PATHS['deepspeak']}/prep/train",
        '--val_cache', f"{DATASET_PATHS['deepspeak']}/prep/test",
        '--stage1_ckpt', stage1_ckpt,
        # 每个 (实验,seed) 独立 ckpt：train_stage2 自动 resume <out_ckpt>.resume，
        # 若共享 ckpt 会串档污染（修复 P1-5）
        '--out_ckpt', f'{output_dir}/ckpt_{exp_id}.pt',
        '--batch', str(batch if batch is not None else cfg.batch_size_stage2),
        '--lr', str(cfg.lr),
        '--seed', str(seed),
        '--log_dir', f'{output_dir}/logs',
        '--exp_id', exp_id,
    ]
    # 训练轮数：仅当用户【显式】传 --epochs 时才覆盖子进程；否则让 cfg.stage2_epochs
    # （即 training_epochs 阶段的搜索值 {40,80,120}）自然生效。
    # 修复 P0-3：原来无条件传 --epochs 会把搜索值覆盖成同一个数，导致该阶段空转。
    if epochs is not None:
        cmd += ['--epochs', str(epochs)]

    # 设置环境变量传递损失权重（论文契约 λ 默认值 0.3/0.1/0.1/0.5，搜索时经此覆盖）
    env = os.environ.copy()
    env['SYNCX_LAMBDA_CORR_LCMA'] = str(cfg.lambda_corr_lcma)
    env['SYNCX_LAMBDA_SHARED'] = str(cfg.lambda_shared)
    env['SYNCX_LAMBDA_ALIGN'] = str(cfg.lambda_align)
    env['SYNCX_LAMBDA_UNIFORM'] = str(cfg.lambda_uniform)
    # 注意：不传 SYNCX_LAMBDA_KIN —— Stage-2 损失不含 L_kin（train_stage2.py:199-207），
    # 传了也是死代码（修复 P1-7）；λ_kin 仅 Stage-1 用。

    print(f"\n{'='*60}")
    print(f"Running: {exp_id}")
    print(f"Params: lambda_corr={cfg.lambda_corr_lcma}, lambda_shared={cfg.lambda_shared}, "
          f"lambda_align={cfg.lambda_align}, lambda_uniform={cfg.lambda_uniform}, lr={cfg.lr}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # 保存日志
    with open(f'{output_dir}/{exp_id}.log', 'w') as f:
        f.write(result.stdout)
        f.write(result.stderr)

    # 自动选优：读 val 指标；无指标（空 val/NaN/未落盘）→ 记 FAILED，不再靠 returncode 冒充 SUCCESS
    m = read_best_metrics(output_dir, exp_id)
    success = result.returncode == 0 and m is not None and m["val_auc"] is not None \
        and float(m["val_auc"]) == float(m["val_auc"])  # 非 NaN
    return {
        'success': success,
        'val_auc': None if m is None else m['val_auc'],
        'val_ap': None if m is None else m['val_ap'],
        'val_eer': None if m is None else m['val_eer'],
    }


def run_stage(stage_name, best_params=None, epochs=None, seeds=(0,), stage1_ckpt="./ckpt/stage1.pt",
              log_root="logs", batch=None):
    """运行指定阶段的搜索。

    seeds: 论文要求 ≥3 seeds 报 mean±std（Ch4:250）；搜索筛选阶段可只 1 seed 加速，
           最终阶段用 --seeds "0 1 2" 取平均。
    log_root: 日志根目录（默认相对 "logs"；云端主流程传 $PERSIST/logs 落盘到持久卷）。
    batch: 覆盖训练 batch（默认用 cfg.batch_size_stage2；冒烟传小值）。
    """
    configs, step = get_search_configs(stage_name)

    output_dir = os.path.join(log_root, f'hyperparameter_search/{stage_name}')
    os.makedirs(output_dir, exist_ok=True)

    results = []

    total = len(configs) * len(seeds)
    n = 0
    for i, combo in enumerate(configs):
        cfg = combo['config']
        params = combo['params']

        # 应用之前阶段的最佳参数
        if best_params:
            for key, val in best_params.items():
                setattr(cfg, key, val)

        for s in seeds:
            n += 1
            # exp_id 带 _seedN 后缀：与 logging_utils.exp_group 的 `_seed\d+$` 剥离正则对齐，
            # 出图端 aggregate_seeds 可直接跨 seed 聚合（修复 P0-4/H1）
            exp_id = f'{stage_name}_{i:03d}_seed{s}'

            r = run_single_experiment(cfg, output_dir, exp_id, epochs=epochs,
                                      stage1_ckpt=stage1_ckpt, seed=s, batch=batch)

            results.append({
                'exp_id': exp_id,
                'epochs': epochs if epochs is not None else cfg.stage2_epochs,
                'params': str(params),
                'seed': s,
                'success': r['success'],
                'val_auc': r['val_auc'],
                'val_ap': r['val_ap'],
                'val_eer': r['val_eer'],
            })

            print(f"[{n}/{total}] {exp_id}: {'SUCCESS' if r['success'] else 'FAILED'}"
                  + (f"  AUC={float(r['val_auc']):.4f}" if r['success'] and r['val_auc'] is not None else ""))

    # 保存汇总（含 val_auc 指标列 → 自动选优依据，修复 P0-1/F1）
    summary_file = f'{output_dir}/summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['exp_id', 'epochs', 'params', 'seed',
                                               'success', 'val_auc', 'val_ap', 'val_eer'])
        writer.writeheader()
        writer.writerows(results)

    # 自动选优打印：按 val_auc 排序（仅成功且有限值的行）
    valid = [r for r in results if r['success'] and r['val_auc'] is not None]
    if valid:
        best = max(valid, key=lambda r: r['val_auc'])
        print(f"\n最佳（按 val_auc）: {best['exp_id']}  AUC={float(best['val_auc']):.4f}  params={best['params']}")
    else:
        print("\n⚠️ 无有效 val_auc（全部 FAILED/NaN）——检查 val_cache 是否有数据、stage1_ckpt 是否就位。")

    print(f"\n{'='*60}")
    print(f"Stage '{stage_name}' complete. Results saved to {summary_file}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Sync-X Hyperparameter Search')
    parser.add_argument('--stage', required=True, choices=['loss_weights', 'learning_rate', 'training_epochs'],
                        help='Search stage to run')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs for screening (20 fast / 40 medium / 80 final). '
                             '⚠️ 运行 --stage training_epochs 时【不要】传本参数：搜索值就是 stage2_epochs '
                             '∈ {40,80,120}，传了会被覆盖成同一个数导致空转。')
    parser.add_argument('--best_lambda_corr', type=float, default=None,
                        help='Best lambda_corr_lcma from previous stage')
    parser.add_argument('--best_lambda_shared', type=float, default=None,
                        help='Best lambda_shared from previous stage')
    parser.add_argument('--best_lambda_align', type=float, default=None,
                        help='Best lambda_align from previous stage')
    parser.add_argument('--best_lambda_uniform', type=float, default=None,
                        help='Best lambda_uniform from previous stage')
    parser.add_argument('--best_lr', type=float, default=None,
                        help='Best lr from previous stage')
    parser.add_argument('--seeds', default='0',
                        help="空格分隔的种子列表，如 '0 1 2'（论文要求 ≥3 seeds mean±std，最终阶段用）")
    parser.add_argument('--stage1_ckpt', default='./ckpt/stage1.pt',
                        help='Stage-1 预训练 ckpt 路径（缺失时 train_stage2 会静默跳过预训练，结果不可比）')
    parser.add_argument('--log_root', default='logs',
                        help='日志根目录（云端主流程传 $PERSIST/logs 落盘到持久卷；默认相对 "logs"）')
    parser.add_argument('--batch', type=int, default=None,
                        help='覆盖训练 batch（默认 cfg.batch_size_stage2；冒烟传小值）')

    args = parser.parse_args()

    # 收集之前阶段的最佳参数
    best_params = {}
    if args.best_lambda_corr is not None:
        best_params['lambda_corr_lcma'] = args.best_lambda_corr
    if args.best_lambda_shared is not None:
        best_params['lambda_shared'] = args.best_lambda_shared
    if args.best_lambda_align is not None:
        best_params['lambda_align'] = args.best_lambda_align
    if args.best_lambda_uniform is not None:
        best_params['lambda_uniform'] = args.best_lambda_uniform
    if args.best_lr is not None:
        best_params['lr'] = args.best_lr

    seeds = tuple(int(s) for s in args.seeds.split())

    print(f"Starting hyperparameter search: stage={args.stage}, epochs={args.epochs or 'default'}, "
          f"seeds={seeds}")
    print(f"Best params from previous stages: {best_params}")

    results = run_stage(args.stage, best_params, epochs=args.epochs, seeds=seeds,
                        stage1_ckpt=args.stage1_ckpt, log_root=args.log_root, batch=args.batch)

    # 打印汇总（含指标）
    print("\n" + "="*60)
    print("SEARCH SUMMARY")
    print("="*60)
    for r in results:
        tag = 'OK' if r['success'] else 'FAIL'
        auc = f"  AUC={float(r['val_auc']):.4f}" if r['success'] and r['val_auc'] is not None else ""
        print(f"  {r['exp_id']}: {r['params']} -> {tag}{auc}")


if __name__ == '__main__':
    main()
