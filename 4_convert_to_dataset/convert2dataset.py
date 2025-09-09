#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 3_labeled_raw_datasets 中的标注CSV转为模型可用的数据集（与模拟生成器结构一致）
- 训练集：所有 _2024154 ~ _2024159.csv
- 测试集：所有 _2024153.csv
- 验证集：默认从训练集中按比例随机划出（默认 10%）

输出：
4_processed_datasets/
  ├─ train.pkl
  ├─ val.pkl
  ├─ test.pkl
  └─ dataset_stats.pkl

依赖：
- pandas, numpy, tqdm
- 本工程已有的 config.Config（用于读取 window_size 以及可选的输出目录）
"""

import os
import re
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 若你的 Config 在别处，请调整导入路径
from config import Config


# ------------ 可调参数（也可直接用 Config） ------------
# 原始数据根目录
RAW_ROOT = Path(r"E:\projects\ML_FP\data\3_labeled_raw_datasets")
# 输出目录（若 Config 中设有 processed_data_dir，会优先使用 Config）
DEFAULT_OUT_DIR = Path(r"E:\projects\ML_FP\data\4_processed_datasets")

# 仅保留的星座前缀（GPS: G，BDS/北斗: C）
ALLOW_CONSTELLATIONS = ("G", "C")

# 从训练集随机划出的验证集比例（设为 0 则不划分）
VAL_SPLIT_RATIO = 0.10

# -----------------------------------------------------


DOY_TEST = "2024153"
DOY_TRAIN_RANGE = set([f"20241{d:02d}" for d in range(54, 60)])  # 2024154~2024159

FNAME_DOY_RE = re.compile(r".*_(\d{7})\.csv$", re.IGNORECASE)


def parse_satellite_id(s: str) -> Tuple[int, str]:
    """
    'G01' -> (1, 'G01')
    'C15' -> (15, 'C15')
    """
    s = s.strip()
    prn = s
    try:
        num = int(s[1:])
    except Exception:
        num = -1
    return num, prn


def row_to_sample(row: pd.Series,
                  history: List[Dict],
                  window_size: int) -> Dict:
    """
    将一行数据（一个历元）+ 历史，拼装为与生成器一致的样本字典
    需要的列：
      time_utc, satellite, rec_x/y/z, sat_x/y/z, S1C, S2W, S2W_S1C_diff, elevation, label
    """
    timestamp = pd.to_datetime(row["time_utc"]).to_pydatetime()

    station_pos = np.array([row["rec_x"], row["rec_y"], row["rec_z"]], dtype=float)
    sat_pos = np.array([row["sat_x"], row["sat_y"], row["sat_z"]], dtype=float)

    s1c = float(row["S1C"])
    s2w = float(row["S2W"])
    diff = float(row["S2W_S1C_diff"]) if pd.notna(row["S2W_S1C_diff"]) else (s2w - s1c)
    elevation = float(row["elevation"]) if pd.notna(row["elevation"]) else np.nan
    label = int(row["label"]) if pd.notna(row["label"]) else 0

    # 序列构造（与生成器一致：历史长度不足时在前端用当前值填充）
    if len(history) >= window_size:
        s2w_seq = [h["s2w_current"] for h in history[-window_size:]]
        s1c_seq = [h["s1c_current"] for h in history[-window_size:]]
        diff_seq = [h["diff_current"] for h in history[-window_size:]]
    else:
        pad = window_size - len(history)
        s2w_seq = [s2w] * pad + [h["s2w_current"] for h in history]
        s1c_seq = [s1c] * pad + [h["s1c_current"] for h in history]
        diff_seq = [diff] * pad + [h["diff_current"] for h in history]

    sat_num, sat_prn = parse_satellite_id(str(row["satellite"]))

    sample = {
        "s2w_current": s2w,
        "s1c_current": s1c,
        "diff_current": diff,
        "s2w_sequence": np.array(s2w_seq, dtype=float),
        "s1c_sequence": np.array(s1c_seq, dtype=float),
        "diff_sequence": np.array(diff_seq, dtype=float),
        "station_position": station_pos,
        "satellite_position": sat_pos,
        "local_time": np.array(
            [timestamp.year, timestamp.month, timestamp.day,
             timestamp.hour, timestamp.minute, timestamp.second],
            dtype=int
        ),
        "satellite_id": sat_num,         # 与生成器一致：G01 -> 1
        "satellite_prn": sat_prn,        # e.g., 'G01'
        "label": label,
        "elevation": elevation,
        "timestamp": timestamp
    }
    return sample


def collect_files(root: Path) -> Dict[str, List[Path]]:
    """
    遍历 root，按 DOY 分组文件列表：
      return {"train": [...], "test": [...]}
    """
    train_files, test_files = [], []
    for p in root.rglob("*.csv"):
        m = FNAME_DOY_RE.match(p.name)
        if not m:
            continue
        doy = m.group(1)
        if doy == DOY_TEST:
            test_files.append(p)
        elif doy in DOY_TRAIN_RANGE:
            train_files.append(p)
        # 其他 DOY（如果有）忽略
    return {"train": sorted(train_files), "test": sorted(test_files)}


def load_and_convert_file(csv_path: Path, window_size: int) -> List[Dict]:
    """
    读取单个CSV并转为样本列表。
    仅处理 GPS(Gxx) / BDS(Cxx)，其他星座文件直接返回空。
    """
    # 从文件名或内容判断卫星
    # 文件名形如 abpo_G01_2024153.csv
    # 也可从列 'satellite' 读取
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["time_utc"],
            dtype={
                "satellite": str, "station": str,
                "rec_x": float, "rec_y": float, "rec_z": float,
                "sat_x": float, "sat_y": float, "sat_z": float,
                "S1C": float, "S2W": float, "S2W_S1C_diff": float,
                "azimuth": float, "elevation": float, "slant_range": float,
                "t_abs": float, "label": int
            }
        )
    except Exception as e:
        print(f"[WARN] 读取失败，跳过: {csv_path} ({e})")
        return []

    # 只保留 GPS/BDS
    # 若整文件的 satellite 都一致可用列首字符判断；如混合则按行判断（理论上你的数据不会混）
    if "satellite" not in df.columns:
        return []

    # 过滤允许的星座
    df = df[df["satellite"].astype(str).str[0].isin(ALLOW_CONSTELLATIONS)]
    if df.empty:
        return []

    # 按时间排序，构造时序
    df = df.sort_values("time_utc").reset_index(drop=True)

    samples: List[Dict] = []
    history: List[Dict] = []
    for _, row in df.iterrows():
        sample = row_to_sample(row, history, window_size)
        samples.append(sample)
        history.append(sample)
        # 限制历史长度（与生成器类似，这里保留最近100个即可，足够窗口使用）
        if len(history) > 100:
            history = history[-100:]

    return samples


def random_split(data: List[Dict], val_ratio: float, seed: int = 42):
    """
    从 data 随机划出 val_ratio 的验证集
    """
    if val_ratio <= 0 or len(data) == 0:
        return data, []

    rng = np.random.RandomState(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    val_size = int(len(data) * val_ratio)
    val_idx = set(idx[:val_size])
    train_idx = [i for i in range(len(data)) if i not in val_idx]
    train_part = [data[i] for i in train_idx]
    val_part = [data[i] for i in idx[:val_size]]
    return train_part, val_part


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[OK] Saved: {path}")


def main():
    config = Config()

    # 读取窗口大小；若未设置则给一个合理默认
    window_size = getattr(getattr(config, "data", object()), "window_size", 16)
    print(f"[Info] window_size = {window_size}")

    # 输出目录优先取 Config.data.processed_data_dir，否则用默认
    out_dir = getattr(getattr(config, "data", object()), "processed_data_dir", None)
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] output dir = {out_dir}")

    # 收集文件
    groups = collect_files(RAW_ROOT)
    train_files = groups["train"]
    test_files = groups["test"]
    print(f"[Info] train files: {len(train_files)}, test files: {len(test_files)}")

    # 转换训练集文件
    train_all: List[Dict] = []
    print("[Step] Converting TRAIN files...")
    for p in tqdm(train_files, desc="TRAIN"):
        train_all.extend(load_and_convert_file(p, window_size))

    # 从训练集随机抽取验证集
    train_final, val_final = random_split(train_all, VAL_SPLIT_RATIO,
                                          seed=getattr(getattr(config, "training", object()), "seed", 42))
    print(f"[Info] train samples: {len(train_final)}, val samples: {len(val_final)}")

    # 转换测试集文件
    test_final: List[Dict] = []
    print("[Step] Converting TEST files...")
    for p in tqdm(test_files, desc="TEST"):
        test_final.extend(load_and_convert_file(p, window_size))
    print(f"[Info] test samples: {len(test_final)}")

    # 统计
    def label_ratio(ds):
        return (sum(int(d.get("label", 0)) for d in ds) / len(ds)) if ds else 0.0

    stats = {
        "total_samples": len(train_final) + len(val_final) + len(test_final),
        "train_samples": len(train_final),
        "val_samples": len(val_final),
        "test_samples": len(test_final),
        "train_flex_power_ratio": label_ratio(train_final),
        "val_flex_power_ratio": label_ratio(val_final),
        "test_flex_power_ratio": label_ratio(test_final),
        "source_root": str(RAW_ROOT),
        "allow_constellations": ALLOW_CONSTELLATIONS,
        "window_size": window_size,
        "train_days": sorted(list(DOY_TRAIN_RANGE)),
        "test_day": DOY_TEST,
    }

    # 保存
    save_pickle(train_final, out_dir / "train.pkl")
    save_pickle(val_final, out_dir / "val.pkl")
    save_pickle(test_final, out_dir / "test.pkl")
    save_pickle(stats, out_dir / "dataset_stats.pkl")

    print("\n[Done] Dataset conversion completed.")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
