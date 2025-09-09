"""
Flex Power Detection Training Script
模型训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import logging
from tensorboardX import SummaryWriter

from config import Config
from model import (
    FlexPowerDetectionModel,
    FocalLoss,
    ContrastiveLoss,
    SmoothnessLoss,
    create_model
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FlexPowerDataset(Dataset):
    """Flex Power数据集类"""

    def __init__(self, data_path: str, config: Config, is_training: bool = True):
        self.config = config
        self.is_training = is_training

        # 加载数据
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

        # 计算标准化参数（仅在训练集上计算）
        if is_training:
            self._compute_normalization_params()

    def _compute_normalization_params(self):
        """计算标准化参数"""
        s2w_values = [d['s2w_current'] for d in self.data]
        s1c_values = [d['s1c_current'] for d in self.data]
        diff_values = [d['diff_current'] for d in self.data]

        self.s2w_mean = np.mean(s2w_values)
        self.s2w_std = np.std(s2w_values)
        self.s1c_mean = np.mean(s1c_values)
        self.s1c_std = np.std(s1c_values)
        self.diff_mean = np.mean(diff_values)
        self.diff_std = np.std(diff_values)

        # 保存标准化参数
        norm_params = {
            's2w_mean': self.s2w_mean, 's2w_std': self.s2w_std,
            's1c_mean': self.s1c_mean, 's1c_std': self.s1c_std,
            'diff_mean': self.diff_mean, 'diff_std': self.diff_std
        }

        norm_path = os.path.join(self.config.data.processed_data_dir, 'normalization_params.pkl')
        with open(norm_path, 'wb') as f:
            pickle.dump(norm_params, f)

        logger.info(f"Normalization parameters saved to {norm_path}")

    def set_normalization_params(self, params: Dict):
        """设置标准化参数（用于验证集和测试集）"""
        self.s2w_mean = params['s2w_mean']
        self.s2w_std = params['s2w_std']
        self.s1c_mean = params['s1c_mean']
        self.s1c_std = params['s1c_std']
        self.diff_mean = params['diff_mean']
        self.diff_std = params['diff_std']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 标准化载噪比
        s2w_current = (sample['s2w_current'] - self.s2w_mean) / self.s2w_std
        s1c_current = (sample['s1c_current'] - self.s1c_mean) / self.s1c_std
        diff_current = (sample['diff_current'] - self.diff_mean) / self.diff_std

        s2w_sequence = (sample['s2w_sequence'] - self.s2w_mean) / self.s2w_std
        s1c_sequence = (sample['s1c_sequence'] - self.s1c_mean) / self.s1c_std
        diff_sequence = (sample['diff_sequence'] - self.diff_mean) / self.diff_std

        # 位置归一化
        station_pos = sample['station_position'] / 1e7
        satellite_pos = sample['satellite_position'] / 1e7

        # ===== 新增：角度特征（用于训练）=====
        # 缺失时返回 nan，最后统一替换为 0
        elev_deg = float(sample.get('elevation', np.nan))
        azim_deg = float(sample.get('azimuth', np.nan))

        # 角度预处理（裁剪/取模）
        elev_rad = np.deg2rad(np.clip(elev_deg, 0.0, 90.0))
        azim_rad = np.deg2rad(np.mod(azim_deg, 360.0))

        # sin/cos 编码（缺失用 0）
        angle_feats = np.array([
            np.sin(elev_rad), np.cos(elev_rad),
            np.sin(azim_rad), np.cos(azim_rad)
        ], dtype=np.float32)
        angle_feats = np.nan_to_num(angle_feats, nan=0.0)

        return {
            's2w_current': torch.tensor(s2w_current, dtype=torch.float32),
            's1c_current': torch.tensor(s1c_current, dtype=torch.float32),
            'diff_current': torch.tensor(diff_current, dtype=torch.float32),
            's2w_sequence': torch.tensor(s2w_sequence, dtype=torch.float32),
            's1c_sequence': torch.tensor(s1c_sequence, dtype=torch.float32),
            'diff_sequence': torch.tensor(diff_sequence, dtype=torch.float32),
            'station_position': torch.tensor(station_pos, dtype=torch.float32),
            'satellite_position': torch.tensor(satellite_pos, dtype=torch.float32),
            'local_time': torch.tensor(sample['local_time'], dtype=torch.float32),
            'satellite_id': torch.tensor(sample['satellite_id'], dtype=torch.long),
            'angle_features': torch.tensor(angle_feats, dtype=torch.float32),  # <— 新增：4 维角度向量
            'label': torch.tensor(sample['label'], dtype=torch.long),
        }

class Trainer:
    """训练器类"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # 创建模型
        self.model = create_model(config).to(self.device)
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # 创建数据加载器
        self._create_data_loaders()

        # 创建优化器
        self._create_optimizer()

        # 创建损失函数
        self._create_loss_functions()

        # 创建学习率调度器
        self._create_lr_scheduler()

        # 初始化训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0

        # 创建TensorBoard writer
        if config.training.use_tensorboard:
            self.writer = SummaryWriter(config.training.log_dir)
        else:
            self.writer = None

    def _create_data_loaders(self):
        """创建数据加载器"""
        # 训练集
        train_dataset = FlexPowerDataset(
            self.config.data.train_data_path,
            self.config,
            is_training=True
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )

        # 验证集
        val_dataset = FlexPowerDataset(
            self.config.data.val_data_path,
            self.config,
            is_training=False
        )

        # 加载标准化参数
        norm_path = os.path.join(self.config.data.processed_data_dir, 'normalization_params.pkl')
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
        val_dataset.set_normalization_params(norm_params)

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )

        logger.info(f"DataLoaders created - Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")

    def _create_optimizer(self):
        """创建优化器"""
        if self.config.training.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

    def _create_loss_functions(self):
        """创建损失函数"""
        # 主损失函数
        if self.config.training.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=self.config.training.focal_loss_alpha,
                gamma=self.config.training.focal_loss_gamma
            )
        else:
            weights = torch.tensor(self.config.training.class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)

        # 辅助损失函数
        if self.config.training.use_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(
                temperature=self.config.training.contrastive_temperature
            )

        if self.config.training.use_smoothness_constraint:
            self.smoothness_loss = SmoothnessLoss()

    def _create_lr_scheduler(self):
        """创建学习率调度器"""
        if self.config.training.lr_scheduler == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.lr_min
            )
        elif self.config.training.lr_scheduler == 'step':
            self.lr_scheduler = StepLR(
                self.optimizer,
                step_size=self.config.training.lr_step_size,
                gamma=self.config.training.lr_gamma
            )
        elif self.config.training.lr_scheduler == 'exponential':
            self.lr_scheduler = ExponentialLR(
                self.optimizer,
                gamma=self.config.training.lr_gamma
            )
        else:
            self.lr_scheduler = None

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.config.training.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            for key in batch:
                batch[key] = batch[key].to(self.device)

            # 前向传播
            logits, features = self.model(batch)
            labels = batch['label']

            # 计算主损失
            loss = self.criterion(logits, labels)

            # 添加辅助损失
            if self.config.training.use_contrastive_loss:
                contrastive_loss = self.contrastive_loss(features, labels)
                loss += self.config.training.contrastive_weight * contrastive_loss

            if self.config.training.use_smoothness_constraint:
                smoothness_loss = self.smoothness_loss(logits)
                loss += self.config.training.smoothness_weight * smoothness_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': total_correct / total_samples
            })

            # 记录到TensorBoard
            if self.writer and batch_idx % self.config.training.log_every_n_steps == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)

        # 计算epoch统计
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 移动数据到设备
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                # 前向传播
                logits, features = self.model(batch)
                labels = batch['label']

                # 计算损失
                loss = self.criterion(logits, labels)

                # 统计
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples

        # 计算更多指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # True Positive, False Positive, True Negative, False Negative
        tp = np.sum((all_predictions == 1) & (all_labels == 1))
        fp = np.sum((all_predictions == 1) & (all_labels == 0))
        tn = np.sum((all_predictions == 0) & (all_labels == 0))
        fn = np.sum((all_predictions == 0) & (all_labels == 1))

        # 防止除零
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'metrics': metrics,
            'config': self.config
        }

        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f'checkpoint_epoch_{self.current_epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.training.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")

        # 删除旧的检查点
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """清理旧的检查点"""
        checkpoint_dir = self.config.training.checkpoint_dir
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # 保留最近的N个检查点
        if len(checkpoints) > self.config.training.keep_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.training.keep_n_checkpoints]:
                os.remove(os.path.join(checkpoint_dir, checkpoint))
                logger.info(f"Removed old checkpoint: {checkpoint}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        # 添加 weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.lr_scheduler and checkpoint['lr_scheduler_state_dict']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['metrics'].get('accuracy', 0.0)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")

    def train(self):
        """完整训练流程"""
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config.experiment_name} v{self.config.version}")

        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.4f}")

            # 验证
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}, "
                       f"Precision: {val_metrics['precision']:.4f}, "
                       f"Recall: {val_metrics['recall']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")

            # 记录到TensorBoard
            if self.writer:
                self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('train/accuracy', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('val/precision', val_metrics['precision'], epoch)
                self.writer.add_scalar('val/recall', val_metrics['recall'], epoch)
                self.writer.add_scalar('val/f1', val_metrics['f1'], epoch)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            # 更新学习率
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # 检查是否是最佳模型
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # 保存检查点
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)

            # 早停
            if self.early_stopping_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info(f"Training completed! Best validation accuracy: {self.best_val_acc:.4f}")

        if self.writer:
            self.writer.close()

def main():
    """主函数"""
    # 加载配置
    config = Config()

    # 创建必要的目录
    os.makedirs(config.data.data_root, exist_ok=True)
    os.makedirs(config.data.raw_data_dir, exist_ok=True)
    os.makedirs(config.data.processed_data_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # 创建训练器
    trainer = Trainer(config)

    # 检查是否有检查点可以恢复
    checkpoint_dir = config.training.checkpoint_dir
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

            response = input(f"Found checkpoint {latest_checkpoint}. Resume training? (y/n): ")
            if response.lower() == 'y':
                trainer.load_checkpoint(latest_checkpoint)

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()