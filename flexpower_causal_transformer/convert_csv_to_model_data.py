"""
CSV to Model Data Converter
将CSV格式的原始数据转换为模型输入格式
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from glob import glob
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from config import Config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVToModelConverter:
	"""CSV数据转换器"""

	def __init__(self, config: Config):
		self.config = config
		self.normalization_params = {}

	def load_csv_files(self, csv_dir: str) -> Dict[str, pd.DataFrame]:
		"""加载所有CSV文件"""
		csv_files = glob(os.path.join(csv_dir, "*.csv"))
		csv_files = [f for f in csv_files if not os.path.basename(f).startswith(('file_index', 'generation_params'))]

		logger.info(f"Found {len(csv_files)} CSV files")

		data_dict = {}
		for csv_file in tqdm(csv_files, desc="Loading CSV files"):
			basename = os.path.basename(csv_file)
			station_sat = basename.replace('.csv', '')

			try:
				df = pd.read_csv(csv_file)
				df['datetime'] = pd.to_datetime(df['datetime'])
				data_dict[station_sat] = df
				logger.debug(f"Loaded {basename}: {len(df)} records")
			except Exception as e:
				logger.error(f"Error loading {basename}: {e}")
				continue

		return data_dict

	def create_time_windows(self, df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
		"""为数据创建时序窗口"""
		if len(df) < window_size:
			logger.warning(f"DataFrame too short ({len(df)}) for window size {window_size}")
			return pd.DataFrame()

		# 按时间排序
		df = df.sort_values('datetime').reset_index(drop=True)

		windowed_data = []

		for i in range(window_size - 1, len(df)):
			# 当前历元数据
			current = df.iloc[i]

			# 前序窗口数据
			window_start = i - window_size + 1
			window_data = df.iloc[window_start:i + 1]

			# 构建时序特征
			s2w_sequence = window_data['s2w_cnr'].values
			s1c_sequence = window_data['s1c_cnr'].values
			diff_sequence = window_data['diff_cnr'].values

			# 构建样本
			sample = {
				# 当前历元数据
				's2w_current': current['s2w_cnr'],
				's1c_current': current['s1c_cnr'],
				'diff_current': current['diff_cnr'],

				# 时序窗口数据
				's2w_sequence': s2w_sequence,
				's1c_sequence': s1c_sequence,
				'diff_sequence': diff_sequence,

				# 位置信息
				'station_position': np.array([current['station_x'], current['station_y'], current['station_z']]),
				'satellite_position': np.array(
					[current['satellite_x'], current['satellite_y'], current['satellite_z']]),

				# 时间信息
				'local_time': np.array([
					current['year'], current['month'], current['day'],
					current['hour'], current['minute'], current['second']
				]),

				# 卫星信息
				'satellite_id': current['satellite_id'],
				'satellite_prn': current['satellite_prn'],

				# 标签
				'label': current['flex_power_active'],

				# 附加信息
				'elevation': current['elevation'],
				'timestamp': current['datetime'],
				'station_id': current['station_id'],
				'epoch': current['epoch'],

				# 数据质量
				'data_quality': current.get('data_quality', 1),
			}

			windowed_data.append(sample)

		return windowed_data

	def filter_data_quality(self, samples: List[Dict]) -> List[Dict]:
		"""过滤数据质量"""
		filtered_samples = []

		for sample in samples:
			# 检查数据质量标志
			if sample.get('data_quality', 1) == 0:
				continue

			# 检查高度角
			if sample.get('elevation', 0) < 10:
				continue

			# 检查载噪比合理性
			if (sample['s2w_current'] < 20 or sample['s2w_current'] > 60 or
					sample['s1c_current'] < 20 or sample['s1c_current'] > 60):
				continue

			# 检查时序数据完整性
			if (len(sample['s2w_sequence']) != self.config.data.window_size or
					len(sample['s1c_sequence']) != self.config.data.window_size or
					len(sample['diff_sequence']) != self.config.data.window_size):
				continue

			# 检查是否有异常值
			if (np.any(np.isnan(sample['s2w_sequence'])) or
					np.any(np.isnan(sample['s1c_sequence'])) or
					np.any(np.isnan(sample['diff_sequence']))):
				continue

			filtered_samples.append(sample)

		return filtered_samples

	def extract_statistical_features(self, samples: List[Dict]) -> List[Dict]:
		"""提取统计特征"""
		if not self.config.preprocess.extract_statistical_features:
			return samples

		for sample in samples:
			# S2W统计特征
			s2w_seq = sample['s2w_sequence']
			sample['s2w_mean'] = np.mean(s2w_seq)
			sample['s2w_std'] = np.std(s2w_seq)
			sample['s2w_max'] = np.max(s2w_seq)
			sample['s2w_min'] = np.min(s2w_seq)
			sample['s2w_median'] = np.median(s2w_seq)

			# S1C统计特征
			s1c_seq = sample['s1c_sequence']
			sample['s1c_mean'] = np.mean(s1c_seq)
			sample['s1c_std'] = np.std(s1c_seq)
			sample['s1c_max'] = np.max(s1c_seq)
			sample['s1c_min'] = np.min(s1c_seq)
			sample['s1c_median'] = np.median(s1c_seq)

			# 差分统计特征
			diff_seq = sample['diff_sequence']
			sample['diff_mean'] = np.mean(diff_seq)
			sample['diff_std'] = np.std(diff_seq)
			sample['diff_max'] = np.max(diff_seq)
			sample['diff_min'] = np.min(diff_seq)
			sample['diff_median'] = np.median(diff_seq)

		return samples

	def extract_trend_features(self, samples: List[Dict]) -> List[Dict]:
		"""提取趋势特征"""
		if not self.config.preprocess.extract_trend_features:
			return samples

		for sample in samples:
			# S2W趋势
			s2w_seq = sample['s2w_sequence']
			s2w_trend = np.polyfit(range(len(s2w_seq)), s2w_seq, 1)[0]
			sample['s2w_trend'] = s2w_trend

			# S1C趋势
			s1c_seq = sample['s1c_sequence']
			s1c_trend = np.polyfit(range(len(s1c_seq)), s1c_seq, 1)[0]
			sample['s1c_trend'] = s1c_trend

			# 差分趋势
			diff_seq = sample['diff_sequence']
			diff_trend = np.polyfit(range(len(diff_seq)), diff_seq, 1)[0]
			sample['diff_trend'] = diff_trend

		return samples

	def extract_diff_features(self, samples: List[Dict]) -> List[Dict]:
		"""提取差分特征"""
		if not self.config.preprocess.extract_diff_features:
			return samples

		for sample in samples:
			# 一阶差分
			s2w_seq = sample['s2w_sequence']
			if len(s2w_seq) > 1:
				s2w_diff1 = np.diff(s2w_seq)
				sample['s2w_diff1_mean'] = np.mean(s2w_diff1)
				sample['s2w_diff1_std'] = np.std(s2w_diff1)
			else:
				sample['s2w_diff1_mean'] = 0
				sample['s2w_diff1_std'] = 0

			s1c_seq = sample['s1c_sequence']
			if len(s1c_seq) > 1:
				s1c_diff1 = np.diff(s1c_seq)
				sample['s1c_diff1_mean'] = np.mean(s1c_diff1)
				sample['s1c_diff1_std'] = np.std(s1c_diff1)
			else:
				sample['s1c_diff1_mean'] = 0
				sample['s1c_diff1_std'] = 0

		return samples

	def remove_outliers(self, samples: List[Dict]) -> List[Dict]:
		"""移除异常值"""
		if not self.config.preprocess.remove_outliers:
			return samples

		threshold = self.config.preprocess.outlier_threshold

		# 收集所有CNR值
		all_s2w = [s['s2w_current'] for s in samples]
		all_s1c = [s['s1c_current'] for s in samples]
		all_diff = [s['diff_current'] for s in samples]

		# 计算统计量
		s2w_mean, s2w_std = np.mean(all_s2w), np.std(all_s2w)
		s1c_mean, s1c_std = np.mean(all_s1c), np.std(all_s1c)
		diff_mean, diff_std = np.mean(all_diff), np.std(all_diff)

		filtered_samples = []

		for sample in samples:
			# 检查是否为异常值
			if (abs(sample['s2w_current'] - s2w_mean) > threshold * s2w_std or
					abs(sample['s1c_current'] - s1c_mean) > threshold * s1c_std or
					abs(sample['diff_current'] - diff_mean) > threshold * diff_std):
				continue

			filtered_samples.append(sample)

		logger.info(f"Removed {len(samples) - len(filtered_samples)} outliers")
		return filtered_samples

	def data_augmentation(self, samples: List[Dict]) -> List[Dict]:
		"""数据增强"""
		if not self.config.preprocess.use_augmentation:
			return samples

		augmented_samples = samples.copy()
		noise_std = self.config.preprocess.noise_std

		for sample in samples:
			# 只对部分样本进行增强
			if np.random.random() < 0.3:
				aug_sample = sample.copy()

				# 添加噪声
				aug_sample['s2w_current'] += np.random.normal(0, noise_std)
				aug_sample['s1c_current'] += np.random.normal(0, noise_std)
				aug_sample['diff_current'] = aug_sample['s2w_current'] - aug_sample['s1c_current']

				# 对时序数据添加噪声
				aug_sample['s2w_sequence'] = sample['s2w_sequence'] + np.random.normal(0, noise_std,
																					   len(sample['s2w_sequence']))
				aug_sample['s1c_sequence'] = sample['s1c_sequence'] + np.random.normal(0, noise_std,
																					   len(sample['s1c_sequence']))
				aug_sample['diff_sequence'] = aug_sample['s2w_sequence'] - aug_sample['s1c_sequence']

				augmented_samples.append(aug_sample)

		logger.info(f"Data augmentation: {len(samples)} -> {len(augmented_samples)} samples")
		return augmented_samples

	def split_data(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
		"""划分训练/验证/测试集"""
		# 随机打乱
		np.random.shuffle(samples)

		n = len(samples)
		train_size = int(n * self.config.data.train_ratio)
		val_size = int(n * self.config.data.val_ratio)

		train_data = samples[:train_size]
		val_data = samples[train_size:train_size + val_size]
		test_data = samples[train_size + val_size:]

		return train_data, val_data, test_data

	def convert_csv_to_model_data(self, csv_dir: str, output_dir: str):
		"""完整的转换流程"""
		logger.info("Starting CSV to model data conversion...")

		# 1. 加载CSV文件
		data_dict = self.load_csv_files(csv_dir)
		if not data_dict:
			logger.error("No CSV files loaded!")
			return

		# 2. 为每个文件创建时序窗口
		all_samples = []

		for station_sat, df in tqdm(data_dict.items(), desc="Creating time windows"):
			logger.info(f"Processing {station_sat}...")
			windowed_samples = self.create_time_windows(df, self.config.data.window_size)

			if windowed_samples:
				logger.info(f"  Created {len(windowed_samples)} windowed samples")
				all_samples.extend(windowed_samples)
			else:
				logger.warning(f"  No valid samples created for {station_sat}")

		logger.info(f"Total samples before filtering: {len(all_samples)}")

		# 3. 数据质量过滤
		all_samples = self.filter_data_quality(all_samples)
		logger.info(f"Samples after quality filtering: {len(all_samples)}")

		# 4. 提取特征
		all_samples = self.extract_statistical_features(all_samples)
		all_samples = self.extract_trend_features(all_samples)
		all_samples = self.extract_diff_features(all_samples)

		# 5. 移除异常值
		all_samples = self.remove_outliers(all_samples)
		logger.info(f"Samples after outlier removal: {len(all_samples)}")

		# 6. 数据增强
		all_samples = self.data_augmentation(all_samples)

		# 7. 数据集划分
		train_data, val_data, test_data = self.split_data(all_samples)

		logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

		# 8. 统计标签分布
		for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
			labels = [d['label'] for d in data]
			flex_ratio = sum(labels) / len(labels) if labels else 0
			logger.info(f"{name} set Flex Power ratio: {flex_ratio:.2%}")

		# 9. 创建输出目录
		os.makedirs(output_dir, exist_ok=True)

		# 10. 保存数据集
		datasets = {
			'train': (train_data, self.config.data.train_data_path),
			'val': (val_data, self.config.data.val_data_path),
			'test': (test_data, self.config.data.test_data_path)
		}

		for name, (data, path) in datasets.items():
			os.makedirs(os.path.dirname(path), exist_ok=True)
			with open(path, 'wb') as f:
				pickle.dump(data, f)
			logger.info(f"Saved {name} data to {path}")

		# 11. 保存转换统计信息
		stats = {
			'conversion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
			'csv_source_dir': csv_dir,
			'csv_files_processed': len(data_dict),
			'total_samples': len(all_samples),
			'train_samples': len(train_data),
			'val_samples': len(val_data),
			'test_samples': len(test_data),
			'window_size': self.config.data.window_size,
			'augmentation_enabled': self.config.preprocess.use_augmentation,
			'outlier_removal_enabled': self.config.preprocess.remove_outliers,
			'overall_flex_ratio': sum([d['label'] for d in all_samples]) / len(all_samples),
		}

		stats_path = os.path.join(output_dir, 'conversion_stats.pkl')
		with open(stats_path, 'wb') as f:
			pickle.dump(stats, f)
		logger.info(f"Conversion statistics saved to {stats_path}")

		logger.info("CSV to model data conversion completed!")

		return stats


def main():
	"""主函数"""
	import argparse

	parser = argparse.ArgumentParser(description='Convert CSV data to model input format')
	parser.add_argument('--csv_dir', type=str, default='data/raw_csv',
						help='Directory containing CSV files')
	parser.add_argument('--output_dir', type=str, default='data/processed',
						help='Output directory for processed data')
	parser.add_argument('--config_path', type=str, default=None,
						help='Path to config file (optional)')

	args = parser.parse_args()

	# 加载配置
	if args.config_path and os.path.exists(args.config_path):
		config = Config.load(args.config_path)
	else:
		config = Config()

	# 检查输入目录
	if not os.path.exists(args.csv_dir):
		logger.error(f"CSV directory not found: {args.csv_dir}")
		logger.info("Please run generate_raw_csv_data.py first to create CSV files")
		return

	# 创建转换器
	converter = CSVToModelConverter(config)

	# 执行转换
	try:
		stats = converter.convert_csv_to_model_data(args.csv_dir, args.output_dir)

		print("\n" + "=" * 50)
		print("CONVERSION SUMMARY")
		print("=" * 50)
		print(f"CSV files processed: {stats['csv_files_processed']}")
		print(f"Total samples: {stats['total_samples']:,}")
		print(f"Train samples: {stats['train_samples']:,}")
		print(f"Val samples: {stats['val_samples']:,}")
		print(f"Test samples: {stats['test_samples']:,}")
		print(f"Overall Flex Power ratio: {stats['overall_flex_ratio']:.2%}")
		print("=" * 50)

	except Exception as e:
		logger.error(f"Conversion failed: {e}")
		raise


if __name__ == "__main__":
	main()