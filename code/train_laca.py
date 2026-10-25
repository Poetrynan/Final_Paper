#!/usr/bin/env python
"""
train_laca.py — 给 train_stage2.py 补一个「LACA 支路模式」命令行入口
=====================================================================
为什么需要这个包装器：
  train_stage2.py 里 laca_mode 只能从 config.py 的默认值 ("both") 读，
  命令行没有 --laca_mode（只有 evaluate.py 有）。而论文要做 LACA 支路消融，
  必须能训出 corr-only / biomech-only 两个模型，所以这里用最小侵入的方式补上：

  原理：train_stage2.py 开头是 `from config import get_config`，也就是说它的
  模块命名空间里有一个叫 get_config 的函数对象。我们在调用 main() 之前，
  把这个名字替换成自己的版本 —— 新版本先调原来的 get_config 拿到 cfg，
  再按环境变量 SYNCX_LACA_MODE 改 cfg.laca_mode。main() 里 `cfg = get_config()`
  拿到的就是改过的配置，其它逻辑一行都不用动。

用法（等价于 train_stage2.py 的全部参数，只是多认一个环境变量）：
    SYNCX_LACA_MODE=corr python train_laca.py \
        --train_cache ... --val_cache ... --stage1_ckpt ... --out_ckpt ...

注意：
  - 三种模式（both / corr / biomech）的模型结构里所有子模块都会被建出来，
    区别只在 forward 走哪条支路；所以 ckpt 不会因为模式不同而加载失败。
  - 但是！没走到的支路权重是随机初始化的。评测时必须带上一模一样的
    --laca_mode，否则会把没训过的支路算进去，数字是垃圾。
"""
import os
import sys

# 保证能 import 到同目录下的 train_stage2 / config（脚本可能从别处被调起）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_stage2


_ORIG_GET_CONFIG = train_stage2.get_config


def _get_config_with_laca_mode(*args, **kwargs):
    """先取默认配置，再按环境变量覆盖 laca_mode，其余字段保持不变。"""
    cfg = _ORIG_GET_CONFIG(*args, **kwargs)
    mode = os.environ.get("SYNCX_LACA_MODE", "").strip()
    if mode in ("both", "corr", "biomech"):
        cfg.laca_mode = mode
        print(f"[train_laca] 覆盖 cfg.laca_mode = {mode}")
    else:
        print("[train_laca] 未设置 SYNCX_LACA_MODE（或值非法），沿用默认 laca_mode")
    return cfg


# 关键一步：打补丁。必须在 main() 被调用之前完成。
train_stage2.get_config = _get_config_with_laca_mode


if __name__ == "__main__":
    train_stage2.main()
