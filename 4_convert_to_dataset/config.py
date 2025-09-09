# Flex Power Detection - Minimal Config for Data Conversion

from dataclasses import dataclass, field

@dataclass
class DataConfig:
    # 转换后数据输出目录（你的目标文件夹）
    processed_data_dir: str = "E:/projects/ML_FP/data/4_raw_dataset"
    # 前序窗口大小（与生成器/转换脚本的序列长度对齐）
    window_size: int = 5

@dataclass
class TrainingConfig:
    # 随机种子（用于划分验证集等需要随机性的步骤）
    seed: int = 42

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# 可选：便于外部直接 import 使用
default_config = Config()
