"""
Flex Power Detection Visualization Tool
可视化工具：绘制S2W时间序列，标注真值和预测结果
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pickle
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from config import Config
from model import create_model
from train import FlexPowerDataset
from torch.utils.data import DataLoader

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10


class FlexPowerVisualizer:
	"""Flex Power可视化器"""

	def __init__(self, config: Config, checkpoint_path: Optional[str] = None):
		self.config = config
		self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')

		# 如果提供了模型路径，加载模型
		if checkpoint_path:
			self.model = create_model(config).to(self.device)
			self._load_model(checkpoint_path)
		else:
			self.model = None

		# 加载测试数据
		self._load_test_data()

	def _load_model(self, checkpoint_path: str):
		"""加载模型"""
		# 添加 weights_only=False
		checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model.eval()
		print(f"Model loaded from {checkpoint_path}")

	def _load_test_data(self):
		"""加载测试数据"""
		# 加载原始测试数据
		with open(self.config.data.test_data_path, 'rb') as f:
			self.raw_data = pickle.load(f)

		# 如果有模型，创建数据加载器进行预测
		if self.model:
			test_dataset = FlexPowerDataset(
				self.config.data.test_data_path,
				self.config,
				is_training=False
			)

			# 加载标准化参数
			norm_path = os.path.join(self.config.data.processed_data_dir, 'normalization_params.pkl')
			with open(norm_path, 'rb') as f:
				norm_params = pickle.load(f)
			test_dataset.set_normalization_params(norm_params)

			self.test_loader = DataLoader(
				test_dataset,
				batch_size=self.config.training.batch_size,
				shuffle=False
			)

			# 获取模型预测
			self._get_predictions()

	def _get_predictions(self):
		"""获取模型预测结果"""
		self.predictions = []
		self.probabilities = []

		self.model.eval()
		with torch.no_grad():
			for batch in self.test_loader:
				for key in batch:
					batch[key] = batch[key].to(self.device)

				logits, _ = self.model(batch)
				probs = torch.softmax(logits, dim=1)
				preds = torch.argmax(logits, dim=1)

				self.predictions.extend(preds.cpu().numpy())
				self.probabilities.extend(probs.cpu().numpy())

		self.predictions = np.array(self.predictions)
		self.probabilities = np.array(self.probabilities)

	def plot_time_series(self, satellite_prn: str = 'G01', station_idx: int = 0,
						 num_points: int = 500, save_path: Optional[str] = None):
		"""
		绘制S2W时间序列图
		- 橙色标注：真值为1（Flex Power开启）
		- 红色标注：模型预测为1
		- 绿色标注：其他点
		"""
		# 筛选特定卫星和测站的数据
		filtered_indices = []
		for i, data in enumerate(self.raw_data):
			if data['satellite_prn'] == satellite_prn:
				# 简单判断测站（实际应用中可能需要更精确的匹配）
				filtered_indices.append(i)

		if len(filtered_indices) == 0:
			print(f"No data found for satellite {satellite_prn}")
			return

		# 限制显示点数
		filtered_indices = filtered_indices[:num_points]

		# 提取数据
		s2w_values = [self.raw_data[i]['s2w_current'] for i in filtered_indices]
		s1c_values = [self.raw_data[i]['s1c_current'] for i in filtered_indices]
		diff_values = [self.raw_data[i]['diff_current'] for i in filtered_indices]
		true_labels = [self.raw_data[i]['label'] for i in filtered_indices]
		timestamps = [self.raw_data[i]['timestamp'] for i in filtered_indices]

		# 获取预测结果（如果有模型）
		if self.model:
			pred_labels = [self.predictions[i] for i in filtered_indices]
			pred_probs = [self.probabilities[i, 1] for i in filtered_indices]
		else:
			pred_labels = [0] * len(filtered_indices)
			pred_probs = [0] * len(filtered_indices)

		# 创建图形
		fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

		# 时间轴
		time_axis = list(range(len(s2w_values)))

		# 绘制S2W载噪比
		ax1 = axes[0]
		for i in range(len(s2w_values)):
			color = self._get_point_color(true_labels[i], pred_labels[i])
			marker = self._get_point_marker(true_labels[i], pred_labels[i])
			ax1.scatter(time_axis[i], s2w_values[i], c=color, s=20, marker=marker, alpha=0.7)

		ax1.plot(time_axis, s2w_values, 'k-', alpha=0.3, linewidth=0.5)
		ax1.set_ylabel('S2W CNR (dB-Hz)', fontsize=11)
		ax1.set_title(f'Flex Power Detection Results - Satellite {satellite_prn}', fontsize=14, fontweight='bold')
		ax1.grid(True, alpha=0.3)

		# 添加Flex Power区域阴影
		self._add_flex_power_regions(ax1, true_labels, time_axis)

		# 绘制S1C载噪比
		ax2 = axes[1]
		ax2.plot(time_axis, s1c_values, 'b-', alpha=0.7, linewidth=1.5, label='S1C')
		ax2.set_ylabel('S1C CNR (dB-Hz)', fontsize=11)
		ax2.grid(True, alpha=0.3)
		ax2.legend(loc='upper right')

		# 绘制差分值
		ax3 = axes[2]
		for i in range(len(diff_values)):
			color = self._get_point_color(true_labels[i], pred_labels[i])
			ax3.scatter(time_axis[i], diff_values[i], c=color, s=15, alpha=0.7)

		ax3.plot(time_axis, diff_values, 'k-', alpha=0.3, linewidth=0.5)
		ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
		ax3.set_ylabel('S2W - S1C (dB)', fontsize=11)
		ax3.set_xlabel('Time Index', fontsize=11)
		ax3.grid(True, alpha=0.3)

		# 添加图例
		legend_elements = [
			mpatches.Patch(color='orange', label='Truth: Flex Power ON'),
			mpatches.Patch(color='red', label='Predicted: Flex Power ON'),
			mpatches.Patch(color='green', label='Normal (No Flex Power)'),
			mpatches.Patch(color='purple', label='Both Truth & Predicted ON')
		]
		ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

		# 添加性能统计
		if self.model:
			correct = sum([1 for t, p in zip(true_labels, pred_labels) if t == p])
			accuracy = correct / len(true_labels)
			tp = sum([1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1])
			fp = sum([1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1])
			fn = sum([1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0])

			precision = tp / (tp + fp) if (tp + fp) > 0 else 0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0

			stats_text = f'Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}'
			fig.text(0.99, 0.01, stats_text, ha='right', fontsize=10,
					 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		plt.tight_layout()

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"Figure saved to {save_path}")
		else:
			plt.show()

	def _get_point_color(self, true_label: int, pred_label: int) -> str:
		"""根据真值和预测值确定点的颜色"""
		if true_label == 1 and pred_label == 1:
			return 'purple'  # 真值和预测都是Flex Power
		elif true_label == 1:
			return 'orange'  # 仅真值是Flex Power
		elif pred_label == 1:
			return 'red'  # 仅预测是Flex Power
		else:
			return 'green'  # 正常状态

	def _get_point_marker(self, true_label: int, pred_label: int) -> str:
		"""根据真值和预测值确定点的形状"""
		if true_label == 1 and pred_label == 1:
			return 's'  # 方形
		elif true_label == 1:
			return '^'  # 上三角
		elif pred_label == 1:
			return 'v'  # 下三角
		else:
			return 'o'  # 圆形

	def _add_flex_power_regions(self, ax, labels: List[int], time_axis: List[int]):
		"""添加Flex Power区域的阴影"""
		in_flex = False
		start_idx = 0

		for i, label in enumerate(labels):
			if label == 1 and not in_flex:
				in_flex = True
				start_idx = i
			elif label == 0 and in_flex:
				in_flex = False
				ax.axvspan(time_axis[start_idx], time_axis[i - 1],
						   alpha=0.1, color='orange')

		# 处理最后一个区域
		if in_flex:
			ax.axvspan(time_axis[start_idx], time_axis[-1],
					   alpha=0.1, color='orange')

	def plot_satellite_comparison(self, num_satellites: int = 4,
								  save_path: Optional[str] = None):
		"""比较多颗卫星的Flex Power检测结果"""
		fig, axes = plt.subplots(num_satellites, 1, figsize=(15, 3 * num_satellites),
								 sharex=True)

		if num_satellites == 1:
			axes = [axes]

		satellite_list = [f'G{i:02d}' for i in range(1, num_satellites + 1)]

		for idx, sat_prn in enumerate(satellite_list):
			ax = axes[idx]

			# 获取该卫星的数据
			sat_indices = [i for i, d in enumerate(self.raw_data)
						   if d['satellite_prn'] == sat_prn][:200]

			if len(sat_indices) == 0:
				ax.text(0.5, 0.5, f'No data for {sat_prn}',
						ha='center', va='center', transform=ax.transAxes)
				ax.set_ylabel(sat_prn)
				continue

			s2w_values = [self.raw_data[i]['s2w_current'] for i in sat_indices]
			true_labels = [self.raw_data[i]['label'] for i in sat_indices]

			if self.model:
				pred_labels = [self.predictions[i] for i in sat_indices]
			else:
				pred_labels = [0] * len(sat_indices)

			time_axis = list(range(len(s2w_values)))

			# 绘制数据
			for i in range(len(s2w_values)):
				color = self._get_point_color(true_labels[i], pred_labels[i])
				ax.scatter(time_axis[i], s2w_values[i], c=color, s=10, alpha=0.7)

			ax.plot(time_axis, s2w_values, 'k-', alpha=0.3, linewidth=0.5)
			ax.set_ylabel(f'{sat_prn}\nCNR (dB-Hz)', fontsize=9)
			ax.grid(True, alpha=0.3)

			# 添加Flex Power区域
			self._add_flex_power_regions(ax, true_labels, time_axis)

		axes[0].set_title('Multi-Satellite Flex Power Detection Comparison',
						  fontsize=14, fontweight='bold')
		axes[-1].set_xlabel('Time Index', fontsize=11)

		# 添加总图例
		legend_elements = [
			mpatches.Patch(color='orange', label='Truth: ON'),
			mpatches.Patch(color='red', label='Pred: ON'),
			mpatches.Patch(color='green', label='OFF'),
			mpatches.Patch(color='purple', label='Both ON')
		]
		fig.legend(handles=legend_elements, loc='upper right',
				   bbox_to_anchor=(0.98, 0.98), fontsize=9)

		plt.tight_layout()

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"Figure saved to {save_path}")
		else:
			plt.show()

	def plot_prediction_confidence(self, save_path: Optional[str] = None):
		"""绘制预测置信度分布"""
		if not self.model:
			print("No model loaded, cannot plot prediction confidence")
			return

		fig, axes = plt.subplots(1, 2, figsize=(12, 5))

		# 获取真实标签
		true_labels = np.array([d['label'] for d in self.raw_data])

		# 分离正负样本的置信度
		pos_probs = self.probabilities[true_labels == 1, 1]
		neg_probs = self.probabilities[true_labels == 0, 1]

		# 绘制直方图
		ax1 = axes[0]
		ax1.hist(neg_probs, bins=30, alpha=0.5, label='No Flex Power', color='green')
		ax1.hist(pos_probs, bins=30, alpha=0.5, label='Flex Power', color='orange')
		ax1.set_xlabel('Predicted Probability of Flex Power', fontsize=11)
		ax1.set_ylabel('Count', fontsize=11)
		ax1.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
		ax1.legend()
		ax1.grid(True, alpha=0.3)

		# 绘制箱线图
		ax2 = axes[1]
		box_data = [neg_probs, pos_probs]
		bp = ax2.boxplot(box_data, labels=['No Flex Power', 'Flex Power'],
						 patch_artist=True)

		colors = ['green', 'orange']
		for patch, color in zip(bp['boxes'], colors):
			patch.set_facecolor(color)
			patch.set_alpha(0.5)

		ax2.set_ylabel('Predicted Probability', fontsize=11)
		ax2.set_title('Confidence by True Label', fontsize=12, fontweight='bold')
		ax2.grid(True, alpha=0.3)

		plt.tight_layout()

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"Figure saved to {save_path}")
		else:
			plt.show()


def main():
	"""主函数"""
	config = Config()

	# 检查是否有训练好的模型
	checkpoint_path = os.path.join(config.training.checkpoint_dir, 'best_model.pth')

	if os.path.exists(checkpoint_path):
		print(f"Loading model from {checkpoint_path}")
		visualizer = FlexPowerVisualizer(config, checkpoint_path)
	else:
		print("No trained model found, visualizing only ground truth")
		visualizer = FlexPowerVisualizer(config)

	# 创建输出目录
	output_dir = "visualization_results"
	os.makedirs(output_dir, exist_ok=True)

	# 绘制单颗卫星的时间序列
	print("Plotting time series for satellite G01...")
	visualizer.plot_time_series(
		satellite_prn='G03',
		num_points=500,
		save_path=os.path.join(output_dir, 'G01_time_series.png')
	)

	# 绘制多颗卫星比较
	print("Plotting multi-satellite comparison...")
	visualizer.plot_satellite_comparison(
		num_satellites=4,
		save_path=os.path.join(output_dir, 'multi_satellite_comparison.png')
	)

	# 绘制置信度分布
	if visualizer.model:
		print("Plotting prediction confidence distribution...")
		visualizer.plot_prediction_confidence(
			save_path=os.path.join(output_dir, 'confidence_distribution.png')
		)

	print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
	main()