"""
Flex Power Detection Model
基于Transformer-CNN混合架构的Flex Power状态检测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict
import numpy as np


class PositionalEncoding(nn.Module):
	"""位置编码模块"""

	def __init__(self, d_model: int, max_len: int = 100):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() *
							 (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe.unsqueeze(0))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.pe[:, :x.size(1)]


class SatelliteEmbedding(nn.Module):
	"""卫星编号嵌入层"""

	def __init__(self, num_satellites: int = 32, embedding_dim: int = 32):
		super().__init__()
		self.embedding = nn.Embedding(num_satellites + 1, embedding_dim)  # +1 for padding
		self.num_satellites = num_satellites

	def forward(self, satellite_ids: torch.Tensor) -> torch.Tensor:
		# satellite_ids: (batch_size,) with values 1-32 for G01-G32
		return self.embedding(satellite_ids)


class PositionEncoder(nn.Module):
	"""空间位置编码器"""

	def __init__(self, input_dim: int = 6, hidden_dims: list = [128, 64, 32],
				 dropout: float = 0.2):
		super().__init__()
		layers = []
		in_dim = input_dim

		for hidden_dim in hidden_dims:
			layers.extend([
				nn.Linear(in_dim, hidden_dim),
				nn.LayerNorm(hidden_dim),
				nn.ReLU(),
				nn.Dropout(dropout)
			])
			in_dim = hidden_dim

		self.encoder = nn.Sequential(*layers)

	def forward(self, station_pos: torch.Tensor, satellite_pos: torch.Tensor) -> torch.Tensor:
		# Concatenate positions
		positions = torch.cat([station_pos, satellite_pos], dim=-1)
		return self.encoder(positions)


class TimeEncoder(nn.Module):
	"""时间特征编码器"""

	def __init__(self, output_dim: int = 16):
		super().__init__()
		self.output_dim = output_dim
		self.linear = nn.Linear(12, output_dim)  # 12 time features

	def forward(self, time_data: torch.Tensor) -> torch.Tensor:
		# time_data: (batch_size, 6) - year, month, day, hour, minute, second
		batch_size = time_data.shape[0]

		# Extract time features
		year = time_data[:, 0]
		month = time_data[:, 1]
		day = time_data[:, 2]
		hour = time_data[:, 3]
		minute = time_data[:, 4]
		second = time_data[:, 5]

		# Create cyclic features
		month_sin = torch.sin(2 * math.pi * month / 12)
		month_cos = torch.cos(2 * math.pi * month / 12)

		day_sin = torch.sin(2 * math.pi * day / 31)
		day_cos = torch.cos(2 * math.pi * day / 31)

		hour_sin = torch.sin(2 * math.pi * hour / 24)
		hour_cos = torch.cos(2 * math.pi * hour / 24)

		minute_sin = torch.sin(2 * math.pi * minute / 60)
		minute_cos = torch.cos(2 * math.pi * minute / 60)

		second_sin = torch.sin(2 * math.pi * second / 60)
		second_cos = torch.cos(2 * math.pi * second / 60)

		# Day of year
		day_of_year = day + (month - 1) * 30.4  # Approximate
		day_of_year_sin = torch.sin(2 * math.pi * day_of_year / 365)
		day_of_year_cos = torch.cos(2 * math.pi * day_of_year / 365)

		# Stack all features
		time_features = torch.stack([
			month_sin, month_cos, day_sin, day_cos,
			hour_sin, hour_cos, minute_sin, minute_cos,
			second_sin, second_cos, day_of_year_sin, day_of_year_cos
		], dim=-1)

		return self.linear(time_features)


class CNNFeatureExtractor(nn.Module):
	"""CNN时序特征提取器"""

	def __init__(self, input_dim: int, channels: list = [64, 128, 256],
				 kernel_sizes: list = [3, 5, 7], dropout: float = 0.2):
		super().__init__()

		self.convs = nn.ModuleList()
		in_channels = input_dim

		for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
			conv_block = nn.Sequential(
				nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
				nn.BatchNorm1d(out_channels),
				nn.ReLU(),
				nn.Dropout(dropout)
			)
			self.convs.append(conv_block)
			in_channels = out_channels

		self.global_pool = nn.AdaptiveAvgPool1d(1)
		self.output_dim = channels[-1]

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch_size, seq_len, features)
		x = x.transpose(1, 2)  # (batch_size, features, seq_len)

		for conv in self.convs:
			x = conv(x)

		# Global pooling
		x = self.global_pool(x).squeeze(-1)  # (batch_size, channels[-1])
		return x


class TransformerFeatureExtractor(nn.Module):
	"""Transformer时序特征提取器"""

	def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 4,
				 n_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.1):
		super().__init__()

		self.input_projection = nn.Linear(input_dim, d_model)
		self.pos_encoding = PositionalEncoding(d_model)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=n_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

		self.output_dim = d_model

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch_size, seq_len, features)
		x = self.input_projection(x)
		x = self.pos_encoding(x)
		x = self.transformer(x)

		# Use mean pooling over sequence
		x = torch.mean(x, dim=1)  # (batch_size, d_model)
		return x


class FlexPowerDetectionModel(nn.Module):
	"""Flex Power检测主模型"""

	def __init__(self, config):
		super().__init__()
		self.config = config

		# 1. Embedding layers
		self.satellite_embedding = SatelliteEmbedding(
			num_satellites=config.data.satellite_num,
			embedding_dim=config.model.satellite_embedding_dim
		)

		self.position_encoder = PositionEncoder(
			input_dim=6,  # 3 for station + 3 for satellite
			hidden_dims=list(config.model.position_encoder_layers),
			dropout=config.model.position_encoder_dropout
		)

		self.time_encoder = TimeEncoder(
			output_dim=config.model.time_encoding_dim
		)

		# 2. Time series feature extractors
		# Input: S2W sequence (5) + S1C sequence (5) + diff sequence (5) = 15 features
		seq_input_dim = 3  # S2W, S1C, diff at each time step

		self.cnn_extractor = CNNFeatureExtractor(
			input_dim=seq_input_dim,
			channels=list(config.model.cnn_channels),
			kernel_sizes=list(config.model.cnn_kernel_sizes),
			dropout=config.model.cnn_dropout
		)

		self.transformer_extractor = TransformerFeatureExtractor(
			input_dim=seq_input_dim,
			d_model=config.model.transformer_d_model,
			n_heads=config.model.transformer_n_heads,
			n_layers=config.model.transformer_n_layers,
			dim_feedforward=config.model.transformer_dim_feedforward,
			dropout=config.model.transformer_dropout
		)

		# 3. Feature fusion layers
		# Calculate total feature dimension
		fusion_input_dim = (
				3 +  # Current S2W, S1C, diff values
				config.model.satellite_embedding_dim +  # Satellite embedding
				config.model.position_encoder_layers[-1] +  # Position encoding
				config.model.time_encoding_dim +  # Time encoding
				self.cnn_extractor.output_dim +  # CNN features
				self.transformer_extractor.output_dim  # Transformer features
		)

		fusion_layers = []
		in_dim = fusion_input_dim

		for hidden_dim in config.model.fusion_layers:
			fusion_layers.extend([
				nn.Linear(in_dim, hidden_dim),
				nn.LayerNorm(hidden_dim),
				nn.ReLU(),
				nn.Dropout(config.model.fusion_dropout)
			])
			in_dim = hidden_dim

		self.fusion_layers = nn.Sequential(*fusion_layers)

		# 4. Classification head
		classifier_layers = []
		in_dim = config.model.fusion_layers[-1]

		for i, hidden_dim in enumerate(config.model.classifier_layers[:-1]):
			classifier_layers.extend([
				nn.Linear(in_dim, hidden_dim),
				nn.ReLU(),
				nn.Dropout(config.model.classifier_dropout)
			])
			in_dim = hidden_dim

		# Final classification layer
		classifier_layers.append(
			nn.Linear(in_dim, config.model.num_classes)
		)

		self.classifier = nn.Sequential(*classifier_layers)

		# Feature vector for contrastive learning
		self.feature_dim = config.model.fusion_layers[-1]

	def extract_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
		"""提取特征向量"""
		# 1. Current epoch features
		s2w_current = batch['s2w_current'].unsqueeze(-1)  # (batch_size, 1)
		s1c_current = batch['s1c_current'].unsqueeze(-1)
		diff_current = batch['diff_current'].unsqueeze(-1)
		current_features = torch.cat([s2w_current, s1c_current, diff_current], dim=-1)

		# 2. Satellite embedding
		satellite_features = self.satellite_embedding(batch['satellite_id'])

		# 3. Position encoding
		position_features = self.position_encoder(
			batch['station_position'],
			batch['satellite_position']
		)

		# 4. Time encoding
		time_features = self.time_encoder(batch['local_time'])

		# 5. Time series features
		# Prepare sequence data: (batch_size, window_size, 3)
		s2w_seq = batch['s2w_sequence'].unsqueeze(-1)
		s1c_seq = batch['s1c_sequence'].unsqueeze(-1)
		diff_seq = batch['diff_sequence'].unsqueeze(-1)
		seq_data = torch.cat([s2w_seq, s1c_seq, diff_seq], dim=-1)

		# Extract CNN features
		cnn_features = self.cnn_extractor(seq_data)

		# Extract Transformer features
		transformer_features = self.transformer_extractor(seq_data)

		# 6. Concatenate all features
		all_features = torch.cat([
			current_features,
			satellite_features,
			position_features,
			time_features,
			cnn_features,
			transformer_features
		], dim=-1)

		# 7. Feature fusion
		fused_features = self.fusion_layers(all_features)

		return fused_features

	def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		前向传播
		Returns:
			logits: (batch_size, num_classes) - 分类logits
			features: (batch_size, feature_dim) - 特征向量(用于对比学习)
		"""
		features = self.extract_features(batch)
		logits = self.classifier(features)

		return logits, features


class FocalLoss(nn.Module):
	"""Focal Loss for addressing class imbalance"""

	def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		ce_loss = F.cross_entropy(inputs, targets, reduction='none')
		p_t = torch.exp(-ce_loss)
		focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

		if self.reduction == 'mean':
			return focal_loss.mean()
		elif self.reduction == 'sum':
			return focal_loss.sum()
		else:
			return focal_loss


class ContrastiveLoss(nn.Module):
	"""对比学习损失"""

	def __init__(self, temperature: float = 0.07):
		super().__init__()
		self.temperature = temperature

	def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# Normalize features
		features = F.normalize(features, p=2, dim=1)

		# Compute similarity matrix
		similarity = torch.matmul(features, features.T) / self.temperature

		# Create mask for positive pairs
		labels = labels.unsqueeze(0)
		mask = (labels == labels.T).float()

		# Exclude diagonal
		mask.fill_diagonal_(0)

		# Compute loss
		exp_sim = torch.exp(similarity)
		log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

		# Mean over positive pairs
		mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
		loss = -mean_log_prob.mean()

		return loss


class SmoothnessLoss(nn.Module):
	"""平滑性约束损失"""

	def __init__(self):
		super().__init__()

	def forward(self, predictions: torch.Tensor, window_size: int = 3) -> torch.Tensor:
		"""
		计算预测结果的时序平滑性损失
		predictions: (batch_size, num_classes) - 按时间顺序排列
		"""
		if predictions.size(0) < 2:
			return torch.tensor(0.0, device=predictions.device)

		# 计算相邻预测的差异
		probs = F.softmax(predictions, dim=1)
		diff = torch.diff(probs, dim=0)
		loss = torch.mean(torch.abs(diff))

		return loss


def create_model(config) -> FlexPowerDetectionModel:
	"""创建模型实例"""
	return FlexPowerDetectionModel(config)


if __name__ == "__main__":
	# 测试模型
	from config import Config

	config = Config()
	model = create_model(config)

	# 创建虚拟输入
	batch_size = 4
	window_size = config.data.window_size

	batch = {
		's2w_current': torch.randn(batch_size),
		's1c_current': torch.randn(batch_size),
		'diff_current': torch.randn(batch_size),
		's2w_sequence': torch.randn(batch_size, window_size),
		's1c_sequence': torch.randn(batch_size, window_size),
		'diff_sequence': torch.randn(batch_size, window_size),
		'station_position': torch.randn(batch_size, 3),
		'satellite_position': torch.randn(batch_size, 3),
		'local_time': torch.randn(batch_size, 6),
		'satellite_id': torch.randint(1, 33, (batch_size,))
	}

	# 前向传播
	logits, features = model(batch)

	print(f"Model output shape: {logits.shape}")
	print(f"Feature shape: {features.shape}")
	print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")