"""
Flex Power Detection Testing Script
模型测试和评估脚本 - Enhanced with complete data export
"""

import os
import torch
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
import logging
from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, f1_score,
	confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import create_model
from train import FlexPowerDataset
from torch.utils.data import DataLoader
import pandas as pd

# 设置日志
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Tester:
	"""测试器类"""

	def __init__(self, config: Config, checkpoint_path: str):
		self.config = config
		self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
		logger.info(f"Using device: {self.device}")

		# 创建模型并加载权重
		self.model = create_model(config).to(self.device)
		self._load_checkpoint(checkpoint_path)

		# 创建测试数据加载器
		self._create_test_loader()

		# 结果存储 - Enhanced with raw data and metadata
		self.results = {
			'predictions': [],
			'labels': [],
			'probabilities': [],
			'raw_data': [],  # 原始数据
			'metadata': []    # 元数据
		}

	def _load_checkpoint(self, checkpoint_path: str):
		"""加载模型检查点"""
		checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model.eval()

		logger.info(f"Model loaded from {checkpoint_path}")
		logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
		logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")

	def _create_test_loader(self):
		"""创建测试数据加载器"""
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
			shuffle=False,
			num_workers=self.config.training.num_workers,
			pin_memory=self.config.training.pin_memory
		)

		# 保存原始数据用于可视化和导出
		with open(self.config.data.test_data_path, 'rb') as f:
			self.raw_test_data = pickle.load(f)

		logger.info(f"Test dataset loaded: {len(test_dataset)} samples")

	def test(self):
		"""执行测试 - Enhanced to collect all data"""
		logger.info("Starting model testing...")

		self.model.eval()

		all_predictions = []
		all_labels = []
		all_probabilities = []
		all_raw_data = []
		all_metadata = []

		# 数据索引计数器
		data_idx = 0

		# 在 Tester.test() 的 with torch.no_grad(): 循环前，插入一次取 batch 的检查
		sample_batch = next(iter(self.test_loader))
		print("angle_features in test batch:", sample_batch['angle_features'].shape,
			  sample_batch['angle_features'].mean().item(), sample_batch['angle_features'].std().item())

		with torch.no_grad():
			for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
				batch_size = batch['label'].size(0)

				# 移动数据到设备
				for key in batch:
					batch[key] = batch[key].to(self.device)

				# 前向传播
				logits, features = self.model(batch)
				labels = batch['label']

				# 计算预测和概率
				probabilities = torch.softmax(logits, dim=1)
				predictions = torch.argmax(logits, dim=1)

				# 收集结果
				batch_predictions = predictions.cpu().numpy()
				batch_labels = labels.cpu().numpy()
				batch_probabilities = probabilities.cpu().numpy()

				all_predictions.extend(batch_predictions)
				all_labels.extend(batch_labels)
				all_probabilities.extend(batch_probabilities)

				# 收集原始数据和元数据
				for i in range(batch_size):
					if data_idx < len(self.raw_test_data):
						raw_sample = self.raw_test_data[data_idx]

						# 构建完整的数据记录
						# 构建完整的数据记录
						data_record = {
							# 原始CNR数据
							's2w_current': raw_sample['s2w_current'],
							's1c_current': raw_sample['s1c_current'],
							'diff_current': raw_sample['diff_current'],
							's2w_sequence': raw_sample['s2w_sequence'],
							's1c_sequence': raw_sample['s1c_sequence'],
							'diff_sequence': raw_sample['diff_sequence'],

							# 位置数据
							'station_position': raw_sample['station_position'],
							'satellite_position': raw_sample['satellite_position'],

							# 时间数据
							'local_time': raw_sample['local_time'],
							'timestamp': raw_sample['timestamp'] if 'timestamp' in raw_sample else None,

							# 卫星信息
							'satellite_id': raw_sample['satellite_id'],
							'satellite_prn': raw_sample['satellite_prn'],

							# 其他信息
							'elevation': raw_sample.get('elevation', None),
							'azimuth': raw_sample.get('azimuth', None),  # <— 新增：方位角

							# 名称（目前数据源没有，先占位；若将来预处理加入可直接填充）
							'station_name': raw_sample.get('station_name', None)  # <— 新增：测站名称占位
						}

						# 元数据
						metadata = {
							'index': data_idx,
							'batch_idx': batch_idx,
							'in_batch_idx': i,
							'true_label': int(batch_labels[i]),
							'predicted_label': int(batch_predictions[i]),
							'prediction_probability': float(batch_probabilities[i, 1]),  # Flex Power开启的概率
							'confidence': float(np.max(batch_probabilities[i])),
							'is_correct': int(batch_labels[i] == batch_predictions[i])
						}

						all_raw_data.append(data_record)
						all_metadata.append(metadata)

					data_idx += 1

		# 保存结果
		self.results['predictions'] = np.array(all_predictions)
		self.results['labels'] = np.array(all_labels)
		self.results['probabilities'] = np.array(all_probabilities)
		self.results['raw_data'] = all_raw_data
		self.results['metadata'] = all_metadata

		logger.info("Testing completed!")
		logger.info(f"Collected {len(all_raw_data)} samples with complete data")

	def evaluate(self) -> Dict:
		"""评估模型性能"""
		predictions = self.results['predictions']
		labels = self.results['labels']
		probabilities = self.results['probabilities'][:, 1]  # Flex Power开启的概率

		# 计算各种指标
		metrics = {
			'accuracy': accuracy_score(labels, predictions),
			'precision': precision_score(labels, predictions),
			'recall': recall_score(labels, predictions),
			'f1': f1_score(labels, predictions),
			'roc_auc': roc_auc_score(labels, probabilities)
		}

		# 混淆矩阵
		cm = confusion_matrix(labels, predictions)
		metrics['confusion_matrix'] = cm.tolist()

		# 分类报告
		report = classification_report(
			labels, predictions,
			target_names=['No Flex Power', 'Flex Power'],
			output_dict=True
		)
		metrics['classification_report'] = report

		# 打印结果
		logger.info("=" * 50)
		logger.info("Model Evaluation Results:")
		logger.info("=" * 50)
		logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
		logger.info(f"Precision: {metrics['precision']:.4f}")
		logger.info(f"Recall:    {metrics['recall']:.4f}")
		logger.info(f"F1 Score:  {metrics['f1']:.4f}")
		logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
		logger.info("-" * 50)
		logger.info("Confusion Matrix:")
		logger.info(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
		logger.info(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
		logger.info("=" * 50)

		return metrics

	def plot_confusion_matrix(self, save_path: str = None):
		"""绘制混淆矩阵"""
		cm = confusion_matrix(self.results['labels'], self.results['predictions'])

		plt.figure(figsize=(8, 6))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
					xticklabels=['No Flex Power', 'Flex Power'],
					yticklabels=['No Flex Power', 'Flex Power'])
		plt.title('Confusion Matrix')
		plt.ylabel('True Label')
		plt.xlabel('Predicted Label')

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			logger.info(f"Confusion matrix saved to {save_path}")
		else:
			plt.show()

		plt.close()

	def plot_roc_curve(self, save_path: str = None):
		"""绘制ROC曲线"""
		fpr, tpr, thresholds = roc_curve(
			self.results['labels'],
			self.results['probabilities'][:, 1]
		)
		roc_auc = roc_auc_score(
			self.results['labels'],
			self.results['probabilities'][:, 1]
		)

		plt.figure(figsize=(8, 6))
		plt.plot(fpr, tpr, color='darkorange', lw=2,
				 label=f'ROC curve (AUC = {roc_auc:.2f})')
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic (ROC) Curve')
		plt.legend(loc="lower right")
		plt.grid(True, alpha=0.3)

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			logger.info(f"ROC curve saved to {save_path}")
		else:
			plt.show()

		plt.close()

	def analyze_errors(self, num_examples: int = 10):
		"""分析错误案例"""
		predictions = self.results['predictions']
		labels = self.results['labels']
		metadata = self.results['metadata']

		# 找出错误的索引
		errors = predictions != labels
		error_indices = np.where(errors)[0]

		if len(error_indices) == 0:
			logger.info("No errors found!")
			return

		logger.info(f"Found {len(error_indices)} errors out of {len(labels)} samples")
		logger.info(f"Error rate: {len(error_indices) / len(labels):.2%}")

		# 分析错误类型
		false_positives = np.sum((predictions == 1) & (labels == 0))
		false_negatives = np.sum((predictions == 0) & (labels == 1))

		logger.info(f"False Positives: {false_positives}")
		logger.info(f"False Negatives: {false_negatives}")

		# 显示一些错误案例
		logger.info("\nError Examples:")
		for i in range(min(num_examples, len(error_indices))):
			idx = error_indices[i]
			sample = self.results['raw_data'][idx]
			meta = metadata[idx]

			logger.info(f"\nExample {i + 1}:")
			logger.info(f"  Satellite: {sample['satellite_prn']}")
			logger.info(f"  True Label: {meta['true_label']} (Flex Power {'ON' if meta['true_label'] else 'OFF'})")
			logger.info(f"  Predicted: {meta['predicted_label']} (Flex Power {'ON' if meta['predicted_label'] else 'OFF'})")
			logger.info(f"  Confidence: {meta['confidence']:.3f}")
			logger.info(f"  S2W CNR: {sample['s2w_current']:.2f}")
			logger.info(f"  S1C CNR: {sample['s1c_current']:.2f}")
			logger.info(f"  Diff CNR: {sample['diff_current']:.2f}")
			if sample['elevation'] is not None:
				logger.info(f"  Elevation: {sample['elevation']:.1f}°")

	def save_results(self, output_dir: str):
		"""保存测试结果（含逐卫星 + 总体指标、可视化图表、报告、MATLAB兼容数据）"""
		import csv

		os.makedirs(output_dir, exist_ok=True)

		# 1) 保存完整的预测结果（包含原始数据和元数据）
		results_path = os.path.join(output_dir, 'test_results.pkl')
		with open(results_path, 'wb') as f:
			pickle.dump(self.results, f)
		logger.info(f"Complete results saved to {results_path}")

		# 2) 保存逐卫星 + 总体评估指标
		eval_all = self.evaluate_per_satellites()

		# JSON
		metrics_path = os.path.join(output_dir, 'evaluation_metrics_by_satellite.json')
		with open(metrics_path, 'w') as f:
			json.dump(eval_all, f, indent=2)
		logger.info(f"Per-satellite + overall metrics saved to {metrics_path}")

		# CSV
		csv_path = os.path.join(output_dir, 'evaluation_metrics_by_satellite.csv')
		with open(csv_path, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([
				'satellite',
				'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
				'tn', 'fp', 'fn', 'tp'
			])
			# per-satellite
			for sat, m in sorted(eval_all['per_satellite'].items()):
				cm = np.array(m['confusion_matrix'])
				tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
				writer.writerow([
					sat,
					m.get('accuracy', None),
					m.get('precision', None),
					m.get('recall', None),
					m.get('f1', None),
					m.get('roc_auc', None),
					tn, fp, fn, tp
				])
			# overall
			M = eval_all['overall']
			cm = np.array(M['confusion_matrix'])
			tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
			writer.writerow([
				'OVERALL',
				M.get('accuracy', None),
				M.get('precision', None),
				M.get('recall', None),
				M.get('f1', None),
				M.get('roc_auc', None),
				tn, fp, fn, tp
			])
		logger.info(f"Per-satellite + overall metrics CSV saved to {csv_path}")

		# 3) 保存总体可视化图表（混淆矩阵 & ROC）
		cm_path = os.path.join(output_dir, 'confusion_matrix.png')
		self.plot_confusion_matrix(cm_path)

		roc_path = os.path.join(output_dir, 'roc_curve.png')
		self.plot_roc_curve(roc_path)

		# 4) 生成测试报告（仍基于总体指标）
		self._generate_report(output_dir, eval_all['overall'])

		# 5) 保存 MATLAB 兼容数据
		self._save_matlab_compatible_data(output_dir)

	def _save_matlab_compatible_data(self, output_dir: str):
		"""保存MATLAB兼容的数据格式"""
		matlab_data = {
			'predictions': self.results['predictions'].tolist(),
			'labels': self.results['labels'].tolist(),
			'probabilities': self.results['probabilities'].tolist(),
			's2w_current': [d['s2w_current'] for d in self.results['raw_data']],
			's1c_current': [d['s1c_current'] for d in self.results['raw_data']],
			'diff_current': [d['diff_current'] for d in self.results['raw_data']],
			'satellite_prn': [d['satellite_prn'] for d in self.results['raw_data']],
			'elevation': [d['elevation'] if d['elevation'] is not None else -999 for d in self.results['raw_data']],
			'azimuth': [d['azimuth'] if d['azimuth'] is not None else -999 for d in self.results['raw_data']],  # 新增
			'metadata': self.results['metadata']
		}

		matlab_path = os.path.join(output_dir, 'matlab_data.pkl')
		with open(matlab_path, 'wb') as f:
			pickle.dump(matlab_data, f)
		logger.info(f"MATLAB-compatible data saved to {matlab_path}")

	def _generate_report(self, output_dir: str, metrics: Dict):
		"""生成测试报告"""
		report_path = os.path.join(output_dir, 'test_report.txt')

		with open(report_path, 'w') as f:
			f.write("=" * 60 + "\n")
			f.write("FLEX POWER DETECTION MODEL TEST REPORT\n")
			f.write("=" * 60 + "\n\n")

			f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write(f"Model: {self.config.experiment_name} v{self.config.version}\n")
			f.write(f"Test Samples: {len(self.results['labels'])}\n")
			f.write(f"Complete Data Records: {len(self.results['raw_data'])}\n\n")

			f.write("-" * 60 + "\n")
			f.write("PERFORMANCE METRICS\n")
			f.write("-" * 60 + "\n")
			f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
			f.write(f"Precision: {metrics['precision']:.4f}\n")
			f.write(f"Recall:    {metrics['recall']:.4f}\n")
			f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
			f.write(f"ROC AUC:   {metrics['roc_auc']:.4f}\n\n")

			f.write("-" * 60 + "\n")
			f.write("CONFUSION MATRIX\n")
			f.write("-" * 60 + "\n")
			cm = np.array(metrics['confusion_matrix'])
			f.write(f"True Negative:  {cm[0, 0]}\n")
			f.write(f"False Positive: {cm[0, 1]}\n")
			f.write(f"False Negative: {cm[1, 0]}\n")
			f.write(f"True Positive:  {cm[1, 1]}\n\n")

			f.write("-" * 60 + "\n")
			f.write("CLASSIFICATION REPORT\n")
			f.write("-" * 60 + "\n")
			for class_name, class_metrics in metrics['classification_report'].items():
				if class_name in ['No Flex Power', 'Flex Power']:
					f.write(f"\n{class_name}:\n")
					f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
					f.write(f"  Recall:    {class_metrics['recall']:.4f}\n")
					f.write(f"  F1-Score:  {class_metrics['f1-score']:.4f}\n")
					f.write(f"  Support:   {class_metrics['support']}\n")

			f.write("\n" + "=" * 60 + "\n")

		logger.info(f"Test report saved to {report_path}")

	def _station_key(self, station_pos, ndigits: int = 0) -> str:
		"""把 station_position 向量转成可读 key，用四舍五入减少重复"""
		try:
			x, y, z = station_pos
			return f"STN_{round(float(x), ndigits)}_{round(float(y), ndigits)}_{round(float(z), ndigits)}"
		except Exception:
			return "STN_UNKNOWN"

	def _build_timeseries_df(self) -> pd.DataFrame:
		"""把 self.results 汇总为 DataFrame，便于分组绘图（优先使用 station_name）"""
		raw = self.results['raw_data']
		meta = self.results['metadata']
		if not raw or not meta:
			raise ValueError("No results available. Run tester.test() first.")

		rows = []
		for r, m in zip(raw, meta):
			ts = r.get('timestamp', None)
			if ts is None:
				lt = r.get('local_time', None)
				if lt is not None and len(lt) >= 6:
					ts = datetime(int(lt[0]), int(lt[1]), int(lt[2]), int(lt[3]), int(lt[4]), int(lt[5]))
			if isinstance(ts, str):
				ts = pd.to_datetime(ts).to_pydatetime()
			if not isinstance(ts, datetime):
				ts = datetime(2000, 1, 1)

			# ✅ 优先用 station_name；没有则退回到坐标 key
			station_name = r.get('station_name', None)
			if station_name is None or str(station_name).strip() == "":
				station_pos = r.get('station_position', [None, None, None])
				station = self._station_key(station_pos, ndigits=0)
			else:
				station = str(station_name)

			sat = r.get('satellite_prn', 'UNK')
			elev = r.get('elevation', None)
			azim = r.get('azimuth', None)  # 新增：方位角

			rows.append({
				'timestamp': ts,
				'date': ts.date(),
				'station': station,
				'satellite_prn': sat,
				's2w': float(r.get('s2w_current', float('nan'))),
				's1c': float(r.get('s1c_current', float('nan'))),
				'diff': float(r.get('diff_current', float('nan'))),
				'elev': float(elev) if elev is not None else np.nan,  # ✅ 统一为 float/NaN，便于判断缺失
				'azim': float(azim) if azim is not None else np.nan,  # 新增：方位角
				'true_label': int(m.get('true_label', 0)),
				'pred_label': int(m.get('predicted_label', 0)),
				'prob_on': float(m.get('prediction_probability', 0.0)),
				'is_correct': bool(m.get('is_correct', 0)),
			})
		df = pd.DataFrame(rows)
		df = df.sort_values('timestamp').reset_index(drop=True)
		return df

	def _plot_status_spans(self, ax, g):
		"""
		在坐标轴 ax 上用半透明矩形覆盖 TP/TN/FP/FN 区间。
		g: 单个 (station, satellite, date) 分组后的 DataFrame（已按时间排序）
		"""
		import numpy as np
		from matplotlib.patches import Patch

		# 计算每个点的状态
		t = g['true_label'].to_numpy().astype(int)
		p = g['pred_label'].to_numpy().astype(int)
		status = np.empty(len(g), dtype=object)
		status[(t == 1) & (p == 1)] = 'TP'
		status[(t == 0) & (p == 0)] = 'TN'
		status[(t == 0) & (p == 1)] = 'FP'
		status[(t == 1) & (p == 0)] = 'FN'

		# 估计采样步长（秒），用于把区间右边界延伸半步，避免空隙
		ts = pd.to_datetime(g['timestamp']).to_numpy()
		if len(ts) >= 2:
			diffs = (ts[1:] - ts[:-1]).astype('timedelta64[s]').astype(float)
			step_sec = np.median(diffs) if np.isfinite(diffs).any() else 30.0
		else:
			step_sec = 30.0
		half_step = pd.to_timedelta(step_sec / 2.0, unit='s')

		# 合并相邻相同状态为区间
		# 利用“状态变化”划分分段
		boundaries = [0]
		for i in range(1, len(status)):
			if status[i] != status[i - 1]:
				boundaries.append(i)
		boundaries.append(len(status))  # 末尾

		# 颜色与图例
		color_map = {
			'TP': '#4CAF50',  # 绿
			'TN': '#90A4AE',  # 灰蓝
			'FP': '#FF9800',  # 橙
			'FN': '#F44336',  # 红
		}
		alpha = 0.20  # 透明度
		used_labels = set()  # 避免重复图例

		for b in range(len(boundaries) - 1):
			i0, i1 = boundaries[b], boundaries[b + 1] - 1
			st = status[i0]
			if st is None:
				continue
			start = pd.to_datetime(g.iloc[i0]['timestamp'])
			end = pd.to_datetime(g.iloc[i1]['timestamp']) + half_step

			ax.axvspan(start, end, color=color_map[st], alpha=alpha, zorder=0,
					   label=st if st not in used_labels else None)
			used_labels.add(st)

	def plot_all_timeseries(self, output_dir: str, draw_s2w_s1c: bool = False):
		"""
		为每个(测站-卫星-单日)绘制时间序列：
		- 背景①：TP/TN/FP/FN 覆盖区间
		- 背景②：灰色表示缺失高度角的时间段
		- 主曲线：S2W CNR
		"""
		os.makedirs(output_dir, exist_ok=True)
		series_dir = os.path.join(output_dir, "series")
		os.makedirs(series_dir, exist_ok=True)

		df = self._build_timeseries_df()
		group_cols = ['station', 'satellite_prn', 'date']

		for (stn, sat, day), g in df.groupby(group_cols):
			g = g.sort_values('timestamp').reset_index(drop=True)

			fig, ax = plt.subplots(figsize=(12, 5))

			# 背景①：TP/TN/FP/FN
			self._plot_status_spans(ax, g)

			# 主曲线：S2W
			ax.plot(g['timestamp'], g['s2w'], linewidth=1.8, label='S2W CNR', color='black', zorder=5)

			if draw_s2w_s1c:
				ax.plot(g['timestamp'], g['s1c'], linewidth=1.0, alpha=0.7, label='S1C CNR', zorder=6)
				ax.plot(g['timestamp'], g['diff'], linewidth=1.0, alpha=0.6, label='Diff (S2W-S1C)', zorder=6)

			ax.set_title(f"{stn} - {sat} - {day}")  # ✅ stn 已是 station_name 或回退key
			ax.set_xlabel("Time (UTC)")
			ax.set_ylabel("CNR")
			ax.grid(True, alpha=0.3)
			ax.margins(y=0.10)

			# 图例去重
			handles, labels = ax.get_legend_handles_labels()
			uniq = {}
			for h, l in zip(handles, labels):
				if l and l not in uniq:
					uniq[l] = h
			ax.legend(uniq.values(), uniq.keys(), loc='upper left', ncol=2)

			# 保存（用测站名称创建文件夹）
			safe_stn = str(stn).replace(':', '_').replace(' ', '_')
			save_dir = os.path.join(series_dir, safe_stn)
			os.makedirs(save_dir, exist_ok=True)
			fname = f"{sat}_{day}.png"
			save_path = os.path.join(save_dir, fname)
			plt.savefig(save_path, dpi=200, bbox_inches='tight')
			plt.close()

		print(f"[OK] Saved all per-station-satellite daily plots under {series_dir}")

	def _compute_metrics(self, y_true, y_pred, y_prob=None):
		"""稳健计算分类指标；当某些指标不可用时返回 None。"""
		from sklearn.metrics import (
			accuracy_score, precision_score, recall_score, f1_score,
			confusion_matrix, classification_report, roc_auc_score
		)
		metrics = {}
		y_true = np.asarray(y_true)
		y_pred = np.asarray(y_pred)

		# 样本数过少或单一类别时的防守
		unique_labels = np.unique(y_true)
		cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
		metrics['confusion_matrix'] = cm.tolist()
		metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
		try:
			metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
			metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
			metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
		except Exception:
			metrics['precision'] = metrics['recall'] = metrics['f1'] = None

		# 分类报告
		try:
			report = classification_report(
				y_true, y_pred,
				target_names=['No Flex Power', 'Flex Power'],
				output_dict=True, zero_division=0
			)
		except Exception:
			report = {}
		metrics['classification_report'] = report

		# AUC（需要 y_prob 且 y_true 同时包含 0/1 才能算）
		if y_prob is not None and len(unique_labels) == 2:
			try:
				y_prob = np.asarray(y_prob)
				metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
			except Exception:
				metrics['roc_auc'] = None
		else:
			metrics['roc_auc'] = None

		return metrics

	def evaluate_per_satellites(self) -> Dict:
		"""
		按卫星PRN分组评估性能，并返回：
		{
		  "per_satellite": { "G01": {...}, "G02": {...}, ... },
		  "overall": {...}
		}
		其中每个 {...} 都是 _compute_metrics 返回的指标字典。
		"""
		preds = self.results['predictions']
		labels = self.results['labels']
		probs = self.results['probabilities'][:, 1] if self.results['probabilities'] is not None else None
		raw = self.results['raw_data']

		# 对齐卫星PRN
		sats = [d['satellite_prn'] for d in raw]
		sats = np.asarray(sats)

		per_sat = {}
		for sat_id in sorted(set(sats)):
			idx = np.where(sats == sat_id)[0]
			y_true = labels[idx]
			y_pred = preds[idx]
			y_prob = probs[idx] if probs is not None else None
			per_sat[sat_id] = self._compute_metrics(y_true, y_pred, y_prob)

		# 总体
		overall = self._compute_metrics(labels, preds, probs)

		# 简要日志
		logger.info("=" * 60)
		logger.info("Per-Satellite Evaluation:")
		for s in sorted(per_sat.keys()):
			m = per_sat[s]
			logger.info(
				f"{s}: acc={m['accuracy']:.4f}, prec={m.get('precision', 0) if m.get('precision') is not None else 'NA'}, "
				f"rec={m.get('recall', 0) if m.get('recall') is not None else 'NA'}, "
				f"f1={m.get('f1', 0) if m.get('f1') is not None else 'NA'}, "
				f"auc={m.get('roc_auc', 'NA')}")
		logger.info("-" * 60)
		logger.info(f"OVERALL: acc={overall['accuracy']:.4f}, prec={overall.get('precision', 'NA')}, "
					f"rec={overall.get('recall', 'NA')}, f1={overall.get('f1', 'NA')}, auc={overall.get('roc_auc', 'NA')}")
		logger.info("=" * 60)

		return {
			"per_satellite": per_sat,
			"overall": overall
		}


def main():
	"""主函数"""
	# 加载配置
	config = Config()

	# 选择检查点
	checkpoint_dir = config.training.checkpoint_dir
	best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

	if not os.path.exists(best_model_path):
		logger.error(f"Best model not found at {best_model_path}")
		logger.info("Please train the model first using train.py")
		return

	# 创建测试器
	tester = Tester(config, best_model_path)

	# 执行测试
	tester.test()

	# 评估结果
	def evaluate(self) -> Dict:
		"""兼容旧接口：返回包含 per_satellite 和 overall 的指标字典。"""
		return self.evaluate_per_satellites()

	# 分析错误
	tester.analyze_errors(num_examples=5)

	# 保存结果
	output_dir = os.path.join("test_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
	tester.save_results(output_dir)

	# 生成“测站-卫星-单日”时间序列图
	tester.plot_all_timeseries(output_dir, draw_s2w_s1c=True)  # 想同时画 S2W/S1C 则设 True

	logger.info(f"\nAll results saved to: {output_dir}")
	logger.info("You can now use the MATLAB script to visualize the results.")


if __name__ == "__main__":
	main()