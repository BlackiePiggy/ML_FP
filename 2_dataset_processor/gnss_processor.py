#!/usr/bin/env python3
"""
增强版GNSS数据处理程序
支持前后一天SP3文件插值、批处理优化、更详细的日志记录
目录适配：
- 输入CSV：递归遍历 ../data/1_cn0_raw/**/<year>/*.csv
- 输出CSV：写入 output_dir/<year>/xxx.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import configparser
from scipy.interpolate import CubicSpline
from pathlib import Path
import logging
import warnings
import multiprocessing as mp

warnings.filterwarnings('ignore')


class EnhancedSP3Parser:
    """增强版SP3文件解析器，支持多文件插值"""

    def __init__(self, sp3_files):
        """
        初始化解析器
        sp3_files: SP3文件路径列表，按时间顺序排列
        """
        self.sp3_files = sp3_files if isinstance(sp3_files, list) else [sp3_files]
        self.satellite_data = {}
        self.parse_all()

    def parse_all(self):
        """解析所有SP3文件"""
        for sp3_file in self.sp3_files:
            self.parse_single_file(sp3_file)

        # 对每颗卫星的数据按时间排序
        for sat_id in self.satellite_data:
            self.satellite_data[sat_id].sort(key=lambda x: x['time'])

    def parse_single_file(self, sp3_file):
        """解析单个SP3文件"""
        current_time = None

        try:
            with open(sp3_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logging.error(f"读取SP3文件失败 {sp3_file}: {e}")
            return

        for line in lines:
            if line.startswith('*'):
                # 时间戳行
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        year = int(parts[1]); month = int(parts[2]); day = int(parts[3])
                        hour = int(parts[4]); minute = int(parts[5]); second = float(parts[6])
                        current_time = datetime(year, month, day, hour, minute, int(second))
                    except ValueError:
                        continue

            elif line.startswith('PG') and current_time:
                # GPS卫星位置行
                try:
                    sat_id = line[1:4]  # G01, G02等
                    x = float(line[4:18]) * 1000  # km转换为m
                    y = float(line[18:32]) * 1000
                    z = float(line[32:46]) * 1000

                    # 检查是否有效（999999.999999表示无效）
                    if abs(x) > 5e7 or abs(y) > 5e7 or abs(z) > 5e7:
                        continue

                    if sat_id not in self.satellite_data:
                        self.satellite_data[sat_id] = []

                    # 避免相同时间重复
                    exists = any(d['time'] == current_time for d in self.satellite_data[sat_id])
                    if not exists:
                        self.satellite_data[sat_id].append({'time': current_time, 'x': x, 'y': y, 'z': z})
                except (ValueError, IndexError):
                    continue

    def get_interpolated_position(self, sat_id, target_time, method='cubic'):
        """
        插值计算指定时间的卫星位置
        method: 'cubic' 或 'linear'
        """
        if sat_id not in self.satellite_data or len(self.satellite_data[sat_id]) < 2:
            return None, None, None

        sat_data = self.satellite_data[sat_id]
        times = [d['time'] for d in sat_data]

        # 转换为秒数（相对于第一个时间点）
        timestamps = np.array([(t - times[0]).total_seconds() for t in times])
        x_vals = np.array([d['x'] for d in sat_data])
        y_vals = np.array([d['y'] for d in sat_data])
        z_vals = np.array([d['z'] for d in sat_data])

        # 目标时间转换
        target_timestamp = (target_time - times[0]).total_seconds()

        # 允许少量外推（±1小时）
        if target_timestamp < timestamps[0] - 3600 or target_timestamp > timestamps[-1] + 3600:
            return None, None, None

        try:
            if method == 'cubic' and len(timestamps) >= 4:
                cs_x = CubicSpline(timestamps, x_vals, extrapolate=True)
                cs_y = CubicSpline(timestamps, y_vals, extrapolate=True)
                cs_z = CubicSpline(timestamps, z_vals, extrapolate=True)
                x = cs_x(target_timestamp); y = cs_y(target_timestamp); z = cs_z(target_timestamp)
            else:
                x = np.interp(target_timestamp, timestamps, x_vals)
                y = np.interp(target_timestamp, timestamps, y_vals)
                z = np.interp(target_timestamp, timestamps, z_vals)

            return float(x), float(y), float(z)

        except Exception as e:
            logging.warning(f"插值失败 {sat_id} at {target_time}: {e}")
            return None, None, None

    def get_satellite_list(self):
        """获取所有卫星ID列表"""
        return list(self.satellite_data.keys())


def calculate_azimuth_elevation_enhanced(rec_x, rec_y, rec_z, sat_x, sat_y, sat_z):
    """增强版方位角和高度角计算，包含更精确的地球模型"""
    # WGS84椭球参数
    a = 6378137.0  # 长半轴
    f = 1.0 / 298.257223563  # 扁率
    e2 = 2 * f - f * f  # 第一偏心率平方

    # 计算测站的经纬度
    p = np.sqrt(rec_x**2 + rec_y**2)
    lon = np.arctan2(rec_y, rec_x)

    # 迭代计算纬度
    lat = np.arctan2(rec_z, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(rec_z, p * (1 - e2 * N / (N + h)))

    # 相对向量
    dx = sat_x - rec_x; dy = sat_y - rec_y; dz = sat_z - rec_z

    # ENU旋转
    sin_lon = np.sin(lon); cos_lon = np.cos(lon)
    sin_lat = np.sin(lat); cos_lat = np.cos(lat)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    # 方位角（0°=北，顺时针为正）
    azimuth = np.degrees(np.arctan2(e, n))
    if azimuth < 0: azimuth += 360.0

    # 高度角
    horizontal_distance = np.sqrt(e**2 + n**2)
    elevation = np.degrees(np.arctan2(u, horizontal_distance))

    # 斜距
    slant_range = np.sqrt(dx**2 + dy**2 + dz**2)

    return azimuth, elevation, slant_range

def process_station_satellite_batch(args):
    """
    批处理单个测站-卫星组合的数据（用于多进程）
    """
    group_key, group_df, sp3_parser, output_dir, config = args
    station, sat = group_key

    # 获取配置参数
    elevation_cutoff = float(config.get('processing', 'elevation_cutoff', fallback=10.0))
    decimal_places = int(config.get('output', 'decimal_places', fallback=6))

    # 去掉测站名称的后缀数字（如 abpo1070 -> abpo）
    # station_name取前4个字母
    station_name = ''.join(station[:4].lower())

    output_data = []
    group_df = group_df.reset_index(drop=True)

    for idx, row in group_df.iterrows():
        time_utc = pd.to_datetime(row['time_utc'])

        # 获取卫星位置
        sat_x, sat_y, sat_z = sp3_parser.get_interpolated_position(sat, time_utc)
        if sat_x is None:
            continue

        # 接收机位置
        rec_x = row['rec_x']; rec_y = row['rec_y']; rec_z = row['rec_z']

        # 计算方位角/高度角/斜距
        azimuth, elevation, slant_range = calculate_azimuth_elevation_enhanced(
            rec_x, rec_y, rec_z, sat_x, sat_y, sat_z
        )

        # 高度角截止
        if elevation < elevation_cutoff:
            continue

        # 计算S2W-S1C差分
        s1c = row['S1C'] if not pd.isna(row['S1C']) else 0
        s2w = row['S2W'] if not pd.isna(row['S2W']) else 0
        s2w_s1c_diff = s2w - s1c if s1c != 0 and s2w != 0 else 0

        # 输出行（⚠️ 已移除历史时序部分）
        output_row = {
            'time_utc': time_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'satellite': sat,
            'station': station_name,
            'rec_x': round(rec_x, decimal_places),
            'rec_y': round(rec_y, decimal_places),
            'rec_z': round(rec_z, decimal_places),
            'sat_x': round(sat_x, decimal_places),
            'sat_y': round(sat_y, decimal_places),
            'sat_z': round(sat_z, decimal_places),
            'S1C': round(s1c, 2),
            'S2W': round(s2w, 2),
            'S2W_S1C_diff': round(s2w_s1c_diff, 2),
            'azimuth': round(azimuth, 2),
            'elevation': round(elevation, 2),
            'slant_range': round(slant_range, 2),
        }

        output_data.append(output_row)

    if not output_data:
        return f"跳过: {station_name}_{sat} (无有效数据)"

    # 创建输出DataFrame
    output_df = pd.DataFrame(output_data)

    # ——基于第一行时间确定年份与 DOY（儒略日，3 位补零）——
    first_dt = pd.to_datetime(output_df['time_utc'].iloc[0])
    year_str = first_dt.strftime('%Y')
    doy_str = f"{first_dt.timetuple().tm_yday:03d}"   # 001~366
    ydoy = f"{year_str}{doy_str}"                     # e.g., 2025207

    # 创建年份子目录并保存
    year_output_dir = os.path.join(output_dir, year_str)
    os.makedirs(year_output_dir, exist_ok=True)

    output_file = os.path.join(year_output_dir, f'{station_name}_{sat}_{ydoy}.csv')
    output_df.to_csv(output_file, index=False)

    return f"已保存: {output_file} ({len(output_data)} 条记录)"


def find_sp3_files(target_date, sp3_dir, use_adjacent=False):
    """
    查找目标日期及前后一天的SP3文件
    """
    sp3_files = []
    dates_to_check = [target_date]
    if use_adjacent:
        dates_to_check = [target_date - timedelta(days=1), target_date, target_date + timedelta(days=1)]

    for date in dates_to_check:
        year = date.year
        doy = date.timetuple().tm_yday

        patterns = [
            f'IGS*_{year:04d}{doy:03d}*_ORB.SP3',
            f'IGS*_{date.strftime("%Y%m%d")}*_ORB.SP3',
            f'igs*_{year:04d}{doy:03d}*.sp3',
            f'*{year:04d}{doy:03d}*.SP3',
        ]

        for pattern in patterns:
            found_files = list(Path(sp3_dir).glob(pattern))
            if found_files:
                sp3_files.extend([str(f) for f in found_files])
                break

    sp3_files = sorted(list(set(sp3_files)))
    return sp3_files


def setup_logging(config):
    """设置日志系统"""
    log_level = config.get('logging', 'log_level', fallback='INFO')
    log_file = config.get('logging', 'log_file', fallback='gnss_processing.log')

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # 清理旧的 handlers，避免重复日志
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(getattr(logging, log_level, logging.INFO))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter); ch.setFormatter(formatter)

    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def main():
    """主程序"""
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')

    # 设置日志
    logger = setup_logging(config)

    # 路径配置
    input_csv_dir = config['paths']['input_csv_dir']        # 建议: ../data/1_cn0_raw
    sp3_dir        = config['paths']['sp3_dir']             # 建议: ../data/sp3
    output_dir     = config['paths']['output_dir']

    # 处理配置
    use_adjacent_sp3 = config.getboolean('processing', 'use_adjacent_sp3', fallback=False)

    # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ——关键修改：递归遍历年份子目录——
    # 兼容两种写法：1) 只一层年份: 2024/*.csv  2) 更深层也OK
    base = Path(input_csv_dir)
    csv_files = sorted(set([*base.glob('*/*.csv'), *base.rglob('*.csv')]))

    if not csv_files:
        logger.error(f"在 {input_csv_dir} 及其子目录中未找到CSV文件")
        return

    logger.info(f"找到 {len(csv_files)} 个CSV文件待处理")

    for csv_file in csv_files:
        logger.info("=" * 60)
        logger.info(f"处理文件: {csv_file}")

        try:
            df = pd.read_csv(csv_file)
            if 'time_utc' not in df.columns:
                logger.warning(f"{csv_file} 缺少 time_utc 列，跳过")
                continue

            df['time_utc'] = pd.to_datetime(df['time_utc'])

            # 取该文件第一条记录的日期用于星历检索（通常同一文件为同一天）
            csv_date = df['time_utc'].iloc[0].date()

            # 查找SP3文件
            sp3_files = find_sp3_files(pd.to_datetime(csv_date), sp3_dir, use_adjacent_sp3)
            if not sp3_files:
                logger.warning(f"未找到日期 {csv_date} 对应的SP3文件")
                continue

            logger.info(f"使用SP3文件: {', '.join([os.path.basename(f) for f in sp3_files])}")

            # 解析SP3
            sp3_parser = EnhancedSP3Parser(sp3_files)
            logger.info(f"成功解析 {len(sp3_parser.get_satellite_list())} 颗卫星的星历数据")

            # 按测站和卫星分组
            if not {'station', 'sat'}.issubset(df.columns):
                logger.warning(f"{csv_file} 缺少必要列 station 或 sat，跳过")
                continue

            grouped = df.groupby(['station', 'sat'])
            groups = list(grouped)
            logger.info(f"共 {len(groups)} 个测站-卫星组合待处理")

            # 多进程参数
            process_args = [(group_key, group_df, sp3_parser, output_dir, config)
                            for group_key, group_df in groups]

            cpu_count = max(mp.cpu_count(), 1)
            pool_size = min(max(cpu_count - 1, 1), len(groups))

            if pool_size > 1:
                logger.info(f"使用 {pool_size} 个进程并行处理")
                with mp.Pool(pool_size) as pool:
                    results = pool.map(process_station_satellite_batch, process_args)
            else:
                results = [process_station_satellite_batch(args) for args in process_args]

            for result in results:
                logger.info(result)

        except Exception as e:
            logger.error(f"处理文件 {csv_file} 时出错: {e}", exc_info=True)
            continue

    logger.info("=" * 60)
    logger.info("所有文件处理完成！")


if __name__ == "__main__":
    main()
