"""
Flex Power Detection Configuration File
配置文件：包含所有模型、训练和数据处理的参数
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    # 抽取出来的根目录（可改成绝对路径 or 通过环境变量覆盖）
    data_root: Path = Path("../data/2")

    # 其它字段由 data_root 推导，初始化后再赋值
    processed_data_dir: Path = field(init=False)
    train_data_path: Path = field(init=False)
    val_data_path: Path = field(init=False)
    test_data_path: Path = field(init=False)

    # 数据参数
    window_size: int = 5
    satellite_num: int = 32

    # 数据分割比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    sampling_rate: int = 30

    def __post_init__(self):
        self.processed_data_dir = self.data_root / "4_raw_dataset"
        self.train_data_path   = self.processed_data_dir / "train.pkl"
        self.val_data_path     = self.processed_data_dir / "val.pkl"
        self.test_data_path    = self.processed_data_dir / "test.pkl"

@dataclass
class ModelConfig:
    """模型相关配置"""
    # Embedding维度
    satellite_embedding_dim: int = 32

    # 位置编码器
    position_encoder_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    position_encoder_dropout: float = 0.2

    # 时间编码
    time_encoding_dim: int = 16

    # CNN参数
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_dropout: float = 0.2

    # Transformer参数
    transformer_d_model: int = 256
    transformer_n_heads: int = 4
    transformer_n_layers: int = 2
    transformer_dropout: float = 0.1
    transformer_dim_feedforward: int = 512

    # 方位角和高度角
    angle_encoding_dim: int = 16

    # 融合层参数
    fusion_layers: List[int] = field(default_factory=lambda: [512, 256])
    fusion_dropout: float = 0.3

    # 分类层参数
    classifier_layers: List[int] = field(default_factory=lambda: [256, 128, 2])
    classifier_dropout: float = 0.2

    # 输出类别
    num_classes: int = 2  # 0: 未开启, 1: 开启flex power

@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 基础参数
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # 学习率调度
    lr_scheduler: str = "cosine"  # "cosine", "step", "exponential"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    lr_min: float = 1e-6

    # 优化器
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    momentum: float = 0.9  # for SGD
    betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))  # for Adam/AdamW

    # 损失函数权重
    class_weights: List[float] = field(default_factory=lambda: [1.0, 3.0])  # 类别权重 [未开启, 开启]
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    use_focal_loss: bool = True

    # 对比学习
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07

    # 正则化
    use_smoothness_constraint: bool = True
    smoothness_weight: float = 0.05

    # 早停
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # 模型保存
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3

    # 日志
    log_dir: str = "./logs"
    log_every_n_steps: int = 10
    use_tensorboard: bool = True

    # 设备
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 4
    pin_memory: bool = True

    # 随机种子
    seed: int = 42

@dataclass
class PreprocessConfig:
    """预处理相关配置"""
    # 标准化方法
    cnr_normalization: str = "zscore"  # "zscore", "minmax", "robust"
    position_normalization: str = "minmax"

    # 特征工程
    extract_statistical_features: bool = True
    extract_trend_features: bool = True
    extract_diff_features: bool = True

    # 统计特征
    statistical_features: List[str] = field(default_factory=lambda: ["mean", "std", "max", "min", "median"])

    # 趋势特征
    trend_window_size: int = 3

    # 数据增强
    use_augmentation: bool = True
    noise_std: float = 0.01
    time_shift_range: int = 2

    # 异常值处理
    remove_outliers: bool = True
    outlier_threshold: float = 4.0  # 标准差倍数

@dataclass
class Config:
    """总配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    # 实验配置
    experiment_name: str = "flex_power_detection"
    version: str = "1.0.0"
    description: str = "Flex Power状态检测模型"

    def save(self, path: str):
        """保存配置到文件"""
        import json
        import dataclasses

        def dict_factory(field_list):
            return {k: v for k, v in field_list if v is not None}

        config_dict = dataclasses.asdict(self, dict_factory=dict_factory)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        """从文件加载配置"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 递归创建dataclass实例
        data_config = DataConfig(**config_dict['data'])
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        preprocess_config = PreprocessConfig(**config_dict['preprocess'])

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            preprocess=preprocess_config,
            experiment_name=config_dict.get('experiment_name', 'flex_power_detection'),
            version=config_dict.get('version', '1.0.0'),
            description=config_dict.get('description', '')
        )

# 创建默认配置实例
default_config = Config()

if __name__ == "__main__":
    # 测试配置保存和加载
    config = Config()

    # 创建必要的目录
    os.makedirs(config.data.processed_data_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(config.training.checkpoint_dir, "config.json")
    config.save(config_path)
    print(f"配置已保存到: {config_path}")

    # 加载配置
    loaded_config = Config.load(config_path)
    print(f"配置已加载: {loaded_config.experiment_name} v{loaded_config.version}")