"""
Flex Power Detection Testing Script
模型测试和评估脚本
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

		# 结果存储
		self.results = {
			'predictions': [],
			'labels': [],
			'probabilities': [],
			'metadata': []
		}

	def _load_checkpoint(self, checkpoint_path: str):
		"""加载模型检查点"""
		# 修改这一行，添加 weights_only=False
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

		# 保存原始数据用于可视化
		with open(self.config.data.test_data_path, 'rb') as f:
			self.raw_test_data = pickle.load(f)

		logger.info(f"Test dataset loaded: {len(test_dataset)} samples")

	def test(self):
		"""执行测试"""
		logger.info("Starting model testing...")

		self.model.eval()

		all_predictions = []
		all_labels = []
		all_probabilities = []

		with torch.no_grad():
			for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
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
				all_predictions.extend(predictions.cpu().numpy())
				all_labels.extend(labels.cpu().numpy())
				all_probabilities.extend(probabilities.cpu().numpy())

		# 保存结果
		self.results['predictions'] = np.array(all_predictions)
		self.results['labels'] = np.array(all_labels)
		self.results['probabilities'] = np.array(all_probabilities)

		logger.info("Testing completed!")

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
			sample = self.raw_test_data[idx]

			logger.info(f"\nExample {i + 1}:")
			logger.info(f"  Satellite: {sample['satellite_prn']}")
			logger.info(f"  True Label: {labels[idx]} (Flex Power {'ON' if labels[idx] else 'OFF'})")
			logger.info(f"  Predicted: {predictions[idx]} (Flex Power {'ON' if predictions[idx] else 'OFF'})")
			logger.info(f"  Confidence: {self.results['probabilities'][idx, predictions[idx]]:.3f}")
			logger.info(f"  S2W CNR: {sample['s2w_current']:.2f}")
			logger.info(f"  S1C CNR: {sample['s1c_current']:.2f}")
			logger.info(f"  Diff CNR: {sample['diff_current']:.2f}")
			logger.info(f"  Elevation: {sample.get('elevation', 'N/A'):.1f}°")

	def save_results(self, output_dir: str):
		"""保存测试结果"""
		os.makedirs(output_dir, exist_ok=True)

		# 保存预测结果
		results_path = os.path.join(output_dir, 'test_results.pkl')
		with open(results_path, 'wb') as f:
			pickle.dump(self.results, f)
		logger.info(f"Results saved to {results_path}")

		# 保存评估指标
		metrics = self.evaluate()
		metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
		with open(metrics_path, 'w') as f:
			json.dump(metrics, f, indent=2)
		logger.info(f"Metrics saved to {metrics_path}")

		# 保存可视化图表
		cm_path = os.path.join(output_dir, 'confusion_matrix.png')
		self.plot_confusion_matrix(cm_path)

		roc_path = os.path.join(output_dir, 'roc_curve.png')
		self.plot_roc_curve(roc_path)

		# 生成测试报告
		self._generate_report(output_dir, metrics)

	def _generate_report(self, output_dir: str, metrics: Dict):
		"""生成测试报告"""
		report_path = os.path.join(output_dir, 'test_report.txt')

		with open(report_path, 'w') as f:
			f.write("=" * 60 + "\n")
			f.write("FLEX POWER DETECTION MODEL TEST REPORT\n")
			f.write("=" * 60 + "\n\n")

			f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write(f"Model: {self.config.experiment_name} v{self.config.version}\n")
			f.write(f"Test Samples: {len(self.results['labels'])}\n\n")

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
	metrics = tester.evaluate()

	# 分析错误
	tester.analyze_errors(num_examples=5)

	# 保存结果
	output_dir = os.path.join("test_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
	tester.save_results(output_dir)

	logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
	main()