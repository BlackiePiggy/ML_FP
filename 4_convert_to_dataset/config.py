# Flex Power Detection - Config (station/day selection inside)

from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    # 原始标注CSV根目录（会递归搜索 *.csv）
    raw_root: str = r"E:/projects/ML_FP/data/3_labeled_raw_datasets"
    # 转换后数据输出目录
    processed_data_dir: str = r"E:/projects/ML_FP/data/4_processed_datasets"
    # 前序窗口大小（序列长度）
    window_size: int = 5
    # 仅保留的星座前缀（GPS: G，BDS: C）
    allow_constellations: List[str] = field(default_factory=lambda: ["G", "C"])

@dataclass
class SplitConfig:
    # 这些测站的**所有 DOY**都作为测试集（满足：cusv 全部进测试）
    always_test_stations: List[str] = field(default_factory=list)
    # 这些 DOY（所有测站）作为测试集（满足：所有站的 2024153 都进测试）
    test_doys: List[str] = field(default_factory=lambda: ["2024153"])

    # 可选：仅处理这些测站（留空=全量）；黑名单优先级低于白名单
    station_whitelist: List[str] = field(default_factory=list)
    station_blacklist: List[str] = field(default_factory=list)

    # 可选：显式指定训练 DOY（留空=非 test 的都进 train）
    train_doys: List[str] = field(default_factory=list)

    # 训练集中再划出验证集比例
    val_split_ratio: float = 0.10

@dataclass
class TrainingConfig:
    # 随机种子（用于验证集划分）
    seed: int = 42

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# 外部直接 import 使用
default_config = Config()
