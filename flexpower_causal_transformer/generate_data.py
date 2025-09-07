"""
Flex Power Detection Data Generator
生成模拟的Flex Power数据用于训练和测试
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
from config import Config


class FlexPowerDataGenerator:
	"""Flex Power数据生成器"""

	def __init__(self, config: Config):
		self.config = config
		np.random.seed(config.training.seed)
		random.seed(config.training.seed)

		# 定义Flex Power区域（简化为经纬度范围）
		self.flex_power_regions = [
			{'lat': (30, 45), 'lon': (-120, -100)},  # 北美区域
			{'lat': (35, 50), 'lon': (120, 140)},  # 东亚区域
			{'lat': (45, 55), 'lon': (5, 20)},  # 欧洲区域
		]

		# 卫星特性参数
		self.satellite_characteristics = self._generate_satellite_characteristics()

		# 测站位置（XYZ坐标）
		self.station_positions = [
			np.array([-2994427.7762, 4951307.2376, 2674497.9713]),  # 示例测站1
			np.array([-1324452.3421, 5332892.1234, 3234521.8765]),  # 示例测站2
			np.array([4234123.5678, -3987654.3210, 3456789.0123]),  # 示例测站3
		]

	def _generate_satellite_characteristics(self) -> Dict:
		"""生成每颗卫星的特性参数"""
		characteristics = {}
		for sat_id in range(1, 33):  # G01 to G32
			characteristics[f'G{sat_id:02d}'] = {
				'base_s2w_cnr': np.random.uniform(35, 40),  # 基础S2W载噪比
				'base_s1c_cnr': np.random.uniform(34, 39),  # 基础S1C载噪比
				'flex_power_gain': np.random.uniform(2, 5),  # Flex Power增益
				'noise_std': np.random.uniform(0.3, 0.8),  # 噪声标准差
			}
		return characteristics

	def _xyz_to_latlon(self, xyz: np.ndarray) -> Tuple[float, float]:
		"""XYZ坐标转换为经纬度（简化版）"""
		x, y, z = xyz
		r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
		lat = np.arcsin(z / r) * 180 / np.pi
		lon = np.arctan2(y, x) * 180 / np.pi
		return lat, lon

	def _is_in_flex_power_region(self, position: np.ndarray) -> bool:
		"""判断位置是否在Flex Power区域内"""
		lat, lon = self._xyz_to_latlon(position)
		for region in self.flex_power_regions:
			if (region['lat'][0] <= lat <= region['lat'][1] and
					region['lon'][0] <= lon <= region['lon'][1]):
				return True
		return False

	def _generate_satellite_trajectory(self, sat_id: str, start_time: datetime,
									   duration_hours: int = 12) -> List[np.ndarray]:
		"""生成卫星轨迹（简化的圆形轨道）"""
		positions = []
		num_points = duration_hours * 120  # 每30秒一个点

		# 轨道参数
		radius = 26560000  # GPS卫星轨道半径（米）
		inclination = np.random.uniform(50, 60) * np.pi / 180  # 轨道倾角
		initial_phase = np.random.uniform(0, 2 * np.pi)

		for i in range(num_points):
			t = i * 30  # 秒
			angle = initial_phase + (2 * np.pi * t) / (12 * 3600)  # 12小时周期

			# 简化的轨道计算
			x = radius * np.cos(angle)
			y = radius * np.sin(angle) * np.cos(inclination)
			z = radius * np.sin(angle) * np.sin(inclination)

			# 添加一些扰动
			x += np.random.normal(0, 1000)
			y += np.random.normal(0, 1000)
			z += np.random.normal(0, 1000)

			positions.append(np.array([x, y, z]))

		return positions

	def _calculate_elevation_angle(self, station_pos: np.ndarray,
								   satellite_pos: np.ndarray) -> float:
		"""计算高度角（简化版）"""
		# 计算相对位置向量
		rel_pos = satellite_pos - station_pos

		# 简化计算：使用向量夹角
		station_norm = station_pos / np.linalg.norm(station_pos)
		rel_norm = rel_pos / np.linalg.norm(rel_pos)

		cos_angle = np.dot(station_norm, rel_norm)
		elevation = np.arcsin(cos_angle) * 180 / np.pi

		# 确保高度角在合理范围内
		elevation = np.clip(elevation, -90, 90)

		return elevation

	def _generate_cnr_value(self, base_cnr: float, elevation: float,
							is_flex_power: bool, flex_gain: float,
							noise_std: float) -> float:
		"""生成载噪比值"""
		# 高度角影响（高度角越高，信号越好）
		elevation_factor = 1.0 + 0.3 * (elevation / 90.0)

		# 基础CNR
		cnr = base_cnr * elevation_factor

		# Flex Power增益
		if is_flex_power:
			cnr += flex_gain

		# 添加噪声
		cnr += np.random.normal(0, noise_std)

		# 限制范围
		cnr = np.clip(cnr, 20, 60)

		return cnr

	def generate_epoch_data(self, station_idx: int, sat_id: str,
							satellite_pos: np.ndarray,
							timestamp: datetime,
							history_data: List[Dict]) -> Dict:
		"""生成单个历元的数据"""

		station_pos = self.station_positions[station_idx]
		sat_char = self.satellite_characteristics[sat_id]

		# 计算高度角
		elevation = self._calculate_elevation_angle(station_pos, satellite_pos)

		# 判断是否在Flex Power区域
		is_flex_power = self._is_in_flex_power_region(satellite_pos) and elevation > 15

		# 生成S2W和S1C载噪比
		s2w_cnr = self._generate_cnr_value(
			sat_char['base_s2w_cnr'], elevation, is_flex_power,
			sat_char['flex_power_gain'], sat_char['noise_std']
		)

		s1c_cnr = self._generate_cnr_value(
			sat_char['base_s1c_cnr'], elevation, False,  # S1C不受Flex Power影响
			0, sat_char['noise_std']
		)

		# 差分值
		diff_cnr = s2w_cnr - s1c_cnr

		# 构建前序窗口数据
		window_size = self.config.data.window_size
		if len(history_data) >= window_size:
			s2w_sequence = [h['s2w_current'] for h in history_data[-window_size:]]
			s1c_sequence = [h['s1c_current'] for h in history_data[-window_size:]]
			diff_sequence = [h['diff_current'] for h in history_data[-window_size:]]
		else:
			# 填充初始数据
			pad_size = window_size - len(history_data)
			s2w_sequence = [s2w_cnr] * pad_size + [h['s2w_current'] for h in history_data]
			s1c_sequence = [s1c_cnr] * pad_size + [h['s1c_current'] for h in history_data]
			diff_sequence = [diff_cnr] * pad_size + [h['diff_current'] for h in history_data]

		# 构建数据字典
		data = {
			's2w_current': s2w_cnr,
			's1c_current': s1c_cnr,
			'diff_current': diff_cnr,
			's2w_sequence': np.array(s2w_sequence),
			's1c_sequence': np.array(s1c_sequence),
			'diff_sequence': np.array(diff_sequence),
			'station_position': station_pos,
			'satellite_position': satellite_pos,
			'local_time': np.array([
				timestamp.year, timestamp.month, timestamp.day,
				timestamp.hour, timestamp.minute, timestamp.second
			]),
			'satellite_id': int(sat_id[1:]),  # G01 -> 1
			'satellite_prn': sat_id,
			'label': int(is_flex_power),
			'elevation': elevation,
			'timestamp': timestamp
		}

		return data

	def generate_dataset(self, num_samples: int = 10000,
						 start_date: str = "2025-04-01") -> List[Dict]:
		"""生成完整数据集"""
		dataset = []
		start_time = datetime.strptime(start_date, "%Y-%m-%d")

		# 为每个测站和卫星生成数据
		samples_per_combination = num_samples // (len(self.station_positions) * 32)

		for station_idx in range(len(self.station_positions)):
			print(f"Generating data for Station {station_idx + 1}...")

			for sat_num in range(1, 33):
				sat_id = f'G{sat_num:02d}'

				# 生成卫星轨迹
				trajectory = self._generate_satellite_trajectory(
					sat_id, start_time, duration_hours=24
				)

				# 生成时序数据
				history_data = []
				current_time = start_time

				for i in range(min(samples_per_combination, len(trajectory))):
					epoch_data = self.generate_epoch_data(
						station_idx, sat_id, trajectory[i],
						current_time, history_data
					)

					dataset.append(epoch_data)
					history_data.append(epoch_data)

					# 更新时间
					current_time += timedelta(seconds=30)

					# 限制历史数据长度
					if len(history_data) > 100:
						history_data = history_data[-100:]

		return dataset

	def add_data_augmentation(self, dataset: List[Dict]) -> List[Dict]:
		"""数据增强"""
		augmented_data = []

		for data in dataset:
			# 原始数据
			augmented_data.append(data.copy())

			if self.config.preprocess.use_augmentation:
				# 添加噪声增强
				if np.random.random() < 0.3:
					aug_data = data.copy()
					noise_std = self.config.preprocess.noise_std

					aug_data['s2w_current'] += np.random.normal(0, noise_std)
					aug_data['s1c_current'] += np.random.normal(0, noise_std)
					aug_data['diff_current'] = aug_data['s2w_current'] - aug_data['s1c_current']

					aug_data['s2w_sequence'] = aug_data['s2w_sequence'] + np.random.normal(0, noise_std, len(
						aug_data['s2w_sequence']))
					aug_data['s1c_sequence'] = aug_data['s1c_sequence'] + np.random.normal(0, noise_std, len(
						aug_data['s1c_sequence']))
					aug_data['diff_sequence'] = aug_data['s2w_sequence'] - aug_data['s1c_sequence']

					augmented_data.append(aug_data)

		return augmented_data

	def split_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
		"""划分训练集、验证集和测试集"""
		np.random.shuffle(dataset)

		n = len(dataset)
		train_size = int(n * self.config.data.train_ratio)
		val_size = int(n * self.config.data.val_ratio)

		train_data = dataset[:train_size]
		val_data = dataset[train_size:train_size + val_size]
		test_data = dataset[train_size + val_size:]

		return train_data, val_data, test_data

	def save_dataset(self, dataset: List[Dict], filepath: str):
		"""保存数据集"""
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		with open(filepath, 'wb') as f:
			pickle.dump(dataset, f)
		print(f"Dataset saved to {filepath}")

	def generate_and_save_all(self):
		"""生成并保存所有数据集"""
		print("Generating Flex Power detection dataset...")

		# 生成原始数据
		dataset = self.generate_dataset(num_samples=10000)
		print(f"Generated {len(dataset)} samples")

		# 数据增强
		dataset = self.add_data_augmentation(dataset)
		print(f"After augmentation: {len(dataset)} samples")

		# 统计标签分布
		labels = [d['label'] for d in dataset]
		flex_power_ratio = sum(labels) / len(labels)
		print(f"Flex Power ratio: {flex_power_ratio:.2%}")

		# 划分数据集
		train_data, val_data, test_data = self.split_dataset(dataset)
		print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

		# 保存数据集
		self.save_dataset(train_data, self.config.data.train_data_path)
		self.save_dataset(val_data, self.config.data.val_data_path)
		self.save_dataset(test_data, self.config.data.test_data_path)

		# 保存数据统计信息
		stats = {
			'total_samples': len(dataset),
			'train_samples': len(train_data),
			'val_samples': len(val_data),
			'test_samples': len(test_data),
			'flex_power_ratio': flex_power_ratio,
			'station_positions': self.station_positions,
			'satellite_characteristics': self.satellite_characteristics,
			'flex_power_regions': self.flex_power_regions
		}

		stats_path = os.path.join(self.config.data.processed_data_dir, 'dataset_stats.pkl')
		with open(stats_path, 'wb') as f:
			pickle.dump(stats, f)
		print(f"Dataset statistics saved to {stats_path}")


def main():
	"""主函数"""
	config = Config()
	generator = FlexPowerDataGenerator(config)
	generator.generate_and_save_all()

	print("\nDataset generation completed!")
	print(f"Data saved in: {config.data.processed_data_dir}")


if __name__ == "__main__":
	main()