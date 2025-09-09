#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 3_labeled_raw_datasets 中的标注CSV转为模型可用的数据集（与模拟生成器结构一致）
— 训练/测试划分、测站/DOY选择，全部由 config.py / config.ini 决定

满足你的划分需求：
- 测试集：cusv 测站的所有 DOY + 所有测站在 DOY=2024153 的数据
- 训练集：其余数据（非 cusv 且 DOY ≠ 2024153）
"""

import re
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import configparser

from config import default_config, Config  # 改造后的 config.py

FNAME_DOY_RE = re.compile(r".*_(\d{7})\.csv$", re.IGNORECASE)

def parse_satellite_id(s: str) -> Tuple[int, str]:
    s = s.strip()
    prn = s
    try:
        num = int(s[1:])
    except Exception:
        num = -1
    return num, prn

def _station_from_fname(p: Path) -> str:
    # e.g., cusv_G01_2024153.csv -> "cusv"
    return p.name.split("_", 1)[0].lower()

def row_to_sample(row: pd.Series, history: List[Dict], window_size: int) -> Dict:
    """
    与生成器一致的样本字典
    需要的列：
      time_utc, satellite, rec_x/y/z, sat_x/y/z, S1C, S2W, S2W_S1C_diff, elevation, azimuth, label
    """
    timestamp = pd.to_datetime(row["time_utc"]).to_pydatetime()

    station_pos = np.array([row["rec_x"], row["rec_y"], row["rec_z"]], dtype=float)
    sat_pos = np.array([row["sat_x"], row["sat_y"], row["sat_z"]], dtype=float)

    s1c = float(row["S1C"])
    s2w = float(row["S2W"])
    diff = float(row["S2W_S1C_diff"]) if pd.notna(row.get("S2W_S1C_diff", np.nan)) else (s2w - s1c)
    elevation = float(row["elevation"]) if pd.notna(row.get("elevation", np.nan)) else np.nan
    azimuth = float(row["azimuth"]) if pd.notna(row.get("azimuth", np.nan)) else np.nan
    label = int(row["label"]) if pd.notna(row.get("label", np.nan)) else 0

    # 历史序列（不足时前端用当前值填充）
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
        "station_name": str(row["station"]),
        "station_position": station_pos,
        "satellite_position": sat_pos,
        "local_time": np.array(
            [timestamp.year, timestamp.month, timestamp.day,
             timestamp.hour, timestamp.minute, timestamp.second],
            dtype=int
        ),
        "satellite_id": sat_num,      # G01 -> 1
        "satellite_prn": sat_prn,     # 'G01'
        "label": label,
        "azimuth": azimuth,
        "elevation": elevation,
        "timestamp": timestamp
    }
    return sample

def _allowed_station(sta: str, split_cfg) -> bool:
    sta = sta.lower()
    wl = [s.lower() for s in split_cfg.station_whitelist] if split_cfg.station_whitelist else []
    bl = [s.lower() for s in split_cfg.station_blacklist] if split_cfg.station_blacklist else []
    if wl:   # 有白名单 -> 只允许白名单
        return sta in wl
    # 无白名单 -> 不在黑名单即可
    return sta not in bl

def _is_test_file(p: Path, split_cfg) -> bool:
    """
    判定文件是否属于测试集：
      1) always_test_stations 中的测站 -> 所有 DOY 都进测试
      2) test_doys 中的 DOY -> 所有测站该 DOY 都进测试
    """
    m = FNAME_DOY_RE.match(p.name)
    if not m:
        return False
    doy = m.group(1)
    sta = _station_from_fname(p)
    ats = set(s.lower() for s in getattr(split_cfg, "always_test_stations", []))
    if sta in ats:
        return True
    if doy in set(split_cfg.test_doys):
        return True
    return False

def collect_files(root: Path, split_cfg) -> Dict[str, List[Path]]:
    train_files, test_files = [], []
    root = Path(root)
    for p in root.rglob("*.csv"):
        sta = _station_from_fname(p)
        if not _allowed_station(sta, split_cfg):
            continue
        if _is_test_file(p, split_cfg):
            test_files.append(p)
        else:
            # 若显式提供 train_doys，则仅收这些 DOY；否则“非 test 都进 train”
            if split_cfg.train_doys:
                m = FNAME_DOY_RE.match(p.name)
                if not m:
                    continue
                doy = m.group(1)
                if doy in set(split_cfg.train_doys):
                    train_files.append(p)
            else:
                train_files.append(p)
    return {"train": sorted(train_files), "test": sorted(test_files)}

def load_and_convert_file(csv_path: Path, window_size: int, allow_constellations: List[str]) -> List[Dict]:
    """
    读取单个CSV并转为样本列表；仅处理 GPS(G*) / BDS(C*)
    """
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

    # 如果缺少 station 列，用文件名前缀填充
    if "station" not in df.columns:
        df["station"] = _station_from_fname(csv_path)

    # 只保留允许的星座
    if "satellite" not in df.columns:
        return []
    df = df[df["satellite"].astype(str).str[0].str.upper().isin([s.upper() for s in allow_constellations])]
    if df.empty:
        return []

    df = df.sort_values("time_utc").reset_index(drop=True)

    samples: List[Dict] = []
    history: List[Dict] = []
    for _, row in df.iterrows():
        sample = row_to_sample(row, history, window_size)
        samples.append(sample)
        history.append(sample)
        # 限制历史长度（足够窗口使用）
        if len(history) > 100:
            history = history[-100:]

    return samples

def random_split(data: List[Dict], val_ratio: float, seed: int = 42):
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
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)  # 可选：protocol=pickle.HIGHEST_PROTOCOL
    print(f"[OK] Saved: {path}")

def _extract_doy(path: Path) -> str:
    m = FNAME_DOY_RE.match(path.name)
    return m.group(1) if m else ""

def _maybe_override_from_ini(cfg: Config, ini_path: Path) -> Config:
    """
    可选：使用 config.ini 覆盖 config.py 的默认值
    """
    if not ini_path.exists():
        return cfg
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")

    # [data]
    if parser.has_section("data"):
        if parser.has_option("data", "raw_root"):
            cfg.data.raw_root = parser.get("data", "raw_root")
        if parser.has_option("data", "processed_data_dir"):
            cfg.data.processed_data_dir = parser.get("data", "processed_data_dir")
        if parser.has_option("data", "window_size"):
            cfg.data.window_size = parser.getint("data", "window_size")
        if parser.has_option("data", "allow_constellations"):
            v = parser.get("data", "allow_constellations")
            cfg.data.allow_constellations = [s.strip() for s in v.split(",") if s.strip()]

    # [split]
    if parser.has_section("split"):
        if parser.has_option("split", "always_test_stations"):
            v = parser.get("split", "always_test_stations")
            cfg.split.always_test_stations = [s.strip().lower() for s in v.split(",") if s.strip()]
        if parser.has_option("split", "test_doys"):
            v = parser.get("split", "test_doys")
            cfg.split.test_doys = [s.strip() for s in v.split(",") if s.strip()]
        if parser.has_option("split", "station_whitelist"):
            v = parser.get("split", "station_whitelist")
            cfg.split.station_whitelist = [s.strip().lower() for s in v.split(",") if s.strip()]
        if parser.has_option("split", "station_blacklist"):
            v = parser.get("split", "station_blacklist")
            cfg.split.station_blacklist = [s.strip().lower() for s in v.split(",") if s.strip()]
        if parser.has_option("split", "train_doys"):
            v = parser.get("split", "train_doys")
            cfg.split.train_doys = [s.strip() for s in v.split(",") if s.strip()]
        if parser.has_option("split", "val_split_ratio"):
            cfg.split.val_split_ratio = parser.getfloat("split", "val_split_ratio")

    # [training]
    if parser.has_section("training"):
        if parser.has_option("training", "seed"):
            cfg.training.seed = parser.getint("training", "seed")

    return cfg

def main():
    # 1) 载入默认配置
    config: Config = default_config

    # 2) 若同目录存在 config.ini，则覆盖（可选）
    ini_path = Path(__file__).with_name("config.ini")
    config = _maybe_override_from_ini(config, ini_path)

    window_size = config.data.window_size
    out_dir = Path(config.data.processed_data_dir)
    raw_root = Path(config.data.raw_root)
    allow_const = config.data.allow_constellations
    val_ratio = config.split.val_split_ratio
    seed = config.training.seed

    print(f"[Info] window_size = {window_size}")
    print(f"[Info] raw_root    = {raw_root}")
    print(f"[Info] output dir  = {out_dir}")
    print(f"[Info] allow const = {allow_const}")
    print(f"[Info] always test stations = {config.split.always_test_stations}")
    print(f"[Info] test DOYs   = {config.split.test_doys}")
    if config.split.station_whitelist:
        print(f"[Info] stations whitelist = {config.split.station_whitelist}")
    if config.split.station_blacklist:
        print(f"[Info] stations blacklist = {config.split.station_blacklist}")
    if config.split.train_doys:
        print(f"[Info] train DOYs (explicit) = {config.split.train_doys}")
    print(f"[Info] val split ratio = {val_ratio} (seed={seed})")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) 收集文件（按配置切分 train/test）
    groups = collect_files(raw_root, config.split)
    train_files = groups["train"]
    test_files = groups["test"]
    print(f"[Info] train files: {len(train_files)}, test files: {len(test_files)}")

    # 4) 转换训练集
    train_all: List[Dict] = []
    print("[Step] Converting TRAIN files...")
    for p in tqdm(train_files, desc="TRAIN"):
        train_all.extend(load_and_convert_file(p, window_size, allow_const))

    # 5) 训练 -> 训练/验证
    train_final, val_final = random_split(train_all, val_ratio, seed=seed)
    print(f"[Info] train samples: {len(train_final)}, val samples: {len(val_final)}")

    # 6) 转换测试集
    test_final: List[Dict] = []
    print("[Step] Converting TEST files...")
    for p in tqdm(test_files, desc="TEST"):
        test_final.extend(load_and_convert_file(p, window_size, allow_const))
    print(f"[Info] test samples: {len(test_final)}")

    # 7) 统计信息
    def label_ratio(ds):
        return (sum(int(d.get("label", 0)) for d in ds) / len(ds)) if ds else 0.0

    train_doys = sorted({_extract_doy(p) for p in train_files})
    test_doys  = sorted({_extract_doy(p) for p in test_files})

    stats = {
        "total_samples": len(train_final) + len(val_final) + len(test_final),
        "train_samples": len(train_final),
        "val_samples": len(val_final),
        "test_samples": len(test_final),
        "train_flex_power_ratio": label_ratio(train_final),
        "val_flex_power_ratio": label_ratio(val_final),
        "test_flex_power_ratio": label_ratio(test_final),
        "source_root": str(raw_root),
        "allow_constellations": allow_const,
        "window_size": window_size,
        "train_doys": train_doys,
        "test_doys": test_doys,
    }

    # 8) 保存
    save_pickle(train_final, out_dir / "train.pkl")
    save_pickle(val_final, out_dir / "val.pkl")
    save_pickle(test_final, out_dir / "test.pkl")
    save_pickle(stats, out_dir / "dataset_stats.pkl")

    print("\n[Done] Dataset conversion completed.")
    print(f"Output dir: {out_dir}")

if __name__ == "__main__":
    main()
