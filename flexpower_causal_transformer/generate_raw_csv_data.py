"""
Raw CSV Data Generator for Flex Power Detection
生成CSV格式的原始数据，按卫星-测站对分文件保存
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
from config import Config


class RawCSVDataGenerator:
	"""原始CSV数据生成器"""

	def __init__(self, config: Config):
		self.config = config
		np.random.seed(config.training.seed)
		random.seed(config.training.seed)

		# 定义Flex Power区域（简化为经纬度范围）
		self.flex_power_regions = [
			{'lat': (30, 45), 'lon': (-120, -100), 'name': 'North_America'},
			{'lat': (35, 50), 'lon': (120, 140), 'name': 'East_Asia'},
			{'lat': (45, 55), 'lon': (5, 20), 'name': 'Europe'},
		]

		# 卫星特性参数
		self.satellite_characteristics = self._generate_satellite_characteristics()

		# 测站信息
		self.stations = [
			{
				'id': 'BJFS', 'name': 'Beijing',
				'position': np.array([-2994427.7762, 4951307.2376, 2674497.9713]),
				'lat': 39.9042, 'lon': 116.4074
			},
			{
				'id': 'SHAO', 'name': 'Shanghai',
				'position': np.array([-1324452.3421, 5332892.1234, 3234521.8765]),
				'lat': 31.2304, 'lon': 121.4737
			},
			{
				'id': 'XIAN', 'name': 'Xian',
				'position': np.array([4234123.5678, -3987654.3210, 3456789.0123]),
				'lat': 34.3416, 'lon': 108.9398
			}
		]

	def _generate_satellite_characteristics(self) -> Dict:
		"""生成每颗卫星的特性参数"""
		characteristics = {}
		for sat_id in range(1, 33):  # G01 to G32
			characteristics[f'G{sat_id:02d}'] = {
				'base_s2w_cnr': np.random.uniform(35, 40),
				'base_s1c_cnr': np.random.uniform(34, 39),
				'flex_power_gain': np.random.uniform(2, 5),
				'noise_std': np.random.uniform(0.3, 0.8),
			}
		return characteristics

	def _xyz_to_latlon(self, xyz: np.ndarray) -> Tuple[float, float]:
		"""XYZ坐标转换为经纬度（简化版）"""
		x, y, z = xyz
		r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
		lat = np.arcsin(z / r) * 180 / np.pi
		lon = np.arctan2(y, x) * 180 / np.pi
		return lat, lon

	def _is_in_flex_power_region(self, position: np.ndarray) -> Tuple[bool, str]:
		"""判断位置是否在Flex Power区域内"""
		lat, lon = self._xyz_to_latlon(position)
		for region in self.flex_power_regions:
			if (region['lat'][0] <= lat <= region['lat'][1] and
					region['lon'][0] <= lon <= region['lon'][1]):
				return True, region['name']
		return False, 'None'

	def _generate_satellite_trajectory(self, sat_id: str, start_time: datetime,
									   duration_hours: int = 12) -> List[Tuple[np.ndarray, datetime]]:
		"""生成卫星轨迹（简化的圆形轨道）"""
		trajectory = []
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

			current_time = start_time + timedelta(seconds=t)
			trajectory.append((np.array([x, y, z]), current_time))

		return trajectory

	def _calculate_elevation_angle(self, station_pos: np.ndarray,
								   satellite_pos: np.ndarray) -> float:
		"""计算高度角（简化版）"""
		rel_pos = satellite_pos - station_pos
		station_norm = station_pos / np.linalg.norm(station_pos)
		rel_norm = rel_pos / np.linalg.norm(rel_pos)

		cos_angle = np.dot(station_norm, rel_norm)
		elevation = np.arcsin(cos_angle) * 180 / np.pi
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

	def generate_station_satellite_data(self, station_info: Dict, sat_id: str,
										start_date: str = "2025-04-01",
										num_epochs: int = 1000) -> pd.DataFrame:
		"""为特定测站-卫星对生成观测数据"""
		start_time = datetime.strptime(start_date, "%Y-%m-%d")

		# 生成卫星轨迹
		trajectory = self._generate_satellite_trajectory(sat_id, start_time, duration_hours=24)

		# 限制观测历元数
		trajectory = trajectory[:num_epochs]

		# 获取卫星特性
		sat_char = self.satellite_characteristics[sat_id]

		data_records = []

		for i, (sat_pos, obs_time) in enumerate(trajectory):
			# 计算高度角
			elevation = self._calculate_elevation_angle(station_info['position'], sat_pos)

			# 跳过低高度角观测
			if elevation < 10:
				continue

			# 判断是否在Flex Power区域
			is_flex_power, flex_region = self._is_in_flex_power_region(sat_pos)
			is_flex_power = is_flex_power and elevation > 15

			# 生成载噪比数据
			s2w_cnr = self._generate_cnr_value(
				sat_char['base_s2w_cnr'], elevation, is_flex_power,
				sat_char['flex_power_gain'], sat_char['noise_std']
			)

			s1c_cnr = self._generate_cnr_value(
				sat_char['base_s1c_cnr'], elevation, False,  # S1C不受Flex Power影响
				0, sat_char['noise_std']
			)

			# 计算差分值
			diff_cnr = s2w_cnr - s1c_cnr

			# 卫星位置（经纬度）
			sat_lat, sat_lon = self._xyz_to_latlon(sat_pos)

			# 构建数据记录
			record = {
				'epoch': i + 1,
				'datetime': obs_time.strftime('%Y-%m-%d %H:%M:%S'),
				'year': obs_time.year,
				'month': obs_time.month,
				'day': obs_time.day,
				'hour': obs_time.hour,
				'minute': obs_time.minute,
				'second': obs_time.second,
				'doy': obs_time.timetuple().tm_yday,  # day of year

				# 测站信息
				'station_id': station_info['id'],
				'station_name': station_info['name'],
				'station_lat': station_info['lat'],
				'station_lon': station_info['lon'],
				'station_x': station_info['position'][0],
				'station_y': station_info['position'][1],
				'station_z': station_info['position'][2],

				# 卫星信息
				'satellite_prn': sat_id,
				'satellite_id': int(sat_id[1:]),  # G01 -> 1
				'satellite_lat': sat_lat,
				'satellite_lon': sat_lon,
				'satellite_x': sat_pos[0],
				'satellite_y': sat_pos[1],
				'satellite_z': sat_pos[2],

				# 观测几何
				'elevation': elevation,
				'azimuth': np.random.uniform(0, 360),  # 简化的方位角

				# 载噪比数据
				's2w_cnr': s2w_cnr,
				's1c_cnr': s1c_cnr,
				'diff_cnr': diff_cnr,

				# Flex Power状态
				'flex_power_active': int(is_flex_power),
				'flex_power_region': flex_region,

				# 质量标志
				'data_quality': 1,  # 1=good, 0=bad
				'multipath_flag': 0,  # 0=no multipath
				'cycle_slip_flag': 0,  # 0=no cycle slip
			}

			data_records.append(record)

		return pd.DataFrame(data_records)

	def generate_all_csv_files(self, output_dir: str = "data/raw_csv",
							   satellites: List[str] = None,
							   num_epochs_per_file: int = 1000):
		"""生成所有测站-卫星对的CSV文件"""

		os.makedirs(output_dir, exist_ok=True)

		if satellites is None:
			satellites = [f'G{i:02d}' for i in range(1, 33)]  # G01-G32

		print(f"Generating CSV files for {len(self.stations)} stations and {len(satellites)} satellites...")

		file_info = []

		for station in self.stations:
			for sat_id in satellites:
				print(f"Processing {station['id']} - {sat_id}...")

				# 生成数据
				df = self.generate_station_satellite_data(
					station, sat_id, num_epochs=num_epochs_per_file
				)

				if len(df) == 0:
					print(f"  No valid observations for {station['id']} - {sat_id}")
					continue

				# 保存CSV文件
				filename = f"{station['id']}_{sat_id}.csv"
				filepath = os.path.join(output_dir, filename)
				df.to_csv(filepath, index=False)

				# 记录文件信息
				flex_power_count = df['flex_power_active'].sum()
				flex_power_ratio = flex_power_count / len(df)

				file_info.append({
					'filename': filename,
					'station_id': station['id'],
					'satellite_prn': sat_id,
					'total_epochs': len(df),
					'flex_power_epochs': flex_power_count,
					'flex_power_ratio': flex_power_ratio,
					'start_time': df['datetime'].iloc[0],
					'end_time': df['datetime'].iloc[-1],
					'mean_elevation': df['elevation'].mean(),
					'mean_s2w_cnr': df['s2w_cnr'].mean(),
					'mean_s1c_cnr': df['s1c_cnr'].mean(),
					'mean_diff_cnr': df['diff_cnr'].mean()
				})

				print(f"  Saved {len(df)} epochs, Flex Power ratio: {flex_power_ratio:.2%}")

		# 保存文件清单和统计信息
		file_info_df = pd.DataFrame(file_info)
		info_path = os.path.join(output_dir, 'file_index.csv')
		file_info_df.to_csv(info_path, index=False)

		# 保存生成参数
		params = {
			'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
			'config_version': self.config.version,
			'num_stations': len(self.stations),
			'num_satellites': len(satellites),
			'num_epochs_per_file': num_epochs_per_file,
			'total_files': len(file_info),
			'flex_power_regions': self.flex_power_regions,
		}

		params_df = pd.DataFrame([params])
		params_path = os.path.join(output_dir, 'generation_params.csv')
		params_df.to_csv(params_path, index=False)

		print(f"\nGeneration completed!")
		print(f"Generated {len(file_info)} files in {output_dir}")
		print(f"File index saved to: {info_path}")
		print(f"Generation parameters saved to: {params_path}")

		# 打印统计摘要
		total_epochs = file_info_df['total_epochs'].sum()
		total_flex_epochs = file_info_df['flex_power_epochs'].sum()
		overall_flex_ratio = total_flex_epochs / total_epochs if total_epochs > 0 else 0

		print(f"\nStatistics Summary:")
		print(f"Total epochs: {total_epochs:,}")
		print(f"Flex Power epochs: {total_flex_epochs:,}")
		print(f"Overall Flex Power ratio: {overall_flex_ratio:.2%}")
		print(f"Average epochs per file: {total_epochs // len(file_info):.0f}")


def main():
	"""主函数"""
	config = Config()
	generator = RawCSVDataGenerator(config)

	# 生成所有CSV文件
	# 可以选择生成部分卫星来减少文件数量
	satellites = [f'G{i:02d}' for i in [1, 2, 3, 15, 20, 25]]  # 选择6颗卫星用于测试

	generator.generate_all_csv_files(
		output_dir="data/raw_csv",
		satellites=satellites,
		num_epochs_per_file=800  # 每个文件800个历元
	)


if __name__ == "__main__":
	main()