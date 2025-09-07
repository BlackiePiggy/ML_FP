import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter


def load_pkl_file(file_path):
	"""加载pkl文件"""
	try:
		with open(file_path, 'rb') as f:
			data = pickle.load(f)
		print(f"成功加载pkl文件: {file_path}")
		print(f"数据类型: {type(data)}")
		return data
	except FileNotFoundError:
		print(f"错误: 找不到文件 {file_path}")
		return None
	except Exception as e:
		print(f"加载文件时出错: {str(e)}")
		return None


def visualize_data(data):
	"""根据数据类型进行可视化"""
	if data is None:
		return

	# 如果是NumPy数组
	if isinstance(data, np.ndarray):
		print(f"数组形状: {data.shape}")

		# 1D数组 - 绘制直方图
		if data.ndim == 1:
			plt.figure(figsize=(10, 6))
			sns.histplot(data, kde=True)
			plt.title('1D数组数据分布')
			plt.xlabel('值')
			plt.ylabel('频率')
			plt.show()

		# 2D数组 - 绘制热图
		elif data.ndim == 2:
			plt.figure(figsize=(10, 8))
			sns.heatmap(data, cmap='viridis')
			plt.title('2D数组热图')
			plt.show()

	# 如果是Pandas DataFrame
	elif isinstance(data, pd.DataFrame):
		print(f"DataFrame形状: {data.shape}")
		print(f"列名: {data.columns.tolist()}")

		# 显示前几行数据
		print("\n数据预览:")
		print(data.head())

		# 绘制数值列的直方图
		plt.figure(figsize=(12, 8))
		data.hist(bins=20, alpha=0.7)
		plt.tight_layout()
		plt.show()

		# 绘制相关性热图（如果有多个数值列）
		numeric_cols = data.select_dtypes(include=[np.number]).columns
		if len(numeric_cols) > 1:
			plt.figure(figsize=(10, 8))
			sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
			plt.title('特征相关性热图')
			plt.show()

	# 如果是字典
	elif isinstance(data, dict):
		print(f"字典包含 {len(data)} 个键值对")
		print(f"键: {list(data.keys())}")

		# 尝试可视化字典中的数值数据
		numeric_items = {k: v for k, v in data.items() if isinstance(v, (int, float))}
		if numeric_items:
			plt.figure(figsize=(10, 6))
			sns.barplot(x=list(numeric_items.keys()), y=list(numeric_items.values()))
			plt.title('字典中的数值数据')
			plt.xticks(rotation=45)
			plt.tight_layout()
			plt.show()

	# 如果是列表
	elif isinstance(data, list):
		print(f"列表包含 {len(data)} 个元素")

		# 尝试统计元素出现频率（如果是可哈希类型）
		try:
			counter = Counter(data)
			if len(counter) <= 20:  # 元素种类不太多时才可视化
				plt.figure(figsize=(10, 6))
				sns.barplot(x=list(counter.keys()), y=list(counter.values()))
				plt.title('列表元素频率分布')
				plt.xticks(rotation=45)
				plt.tight_layout()
				plt.show()
			else:
				print("列表元素种类过多，不适合可视化")
		except TypeError:
			print("列表包含不可哈希元素，无法统计频率")

	# 其他类型
	else:
		print(f"不支持的数据类型 {type(data)} 的可视化")


if __name__ == "__main__":
	# 替换为你的pkl文件路径
	pkl_file_path = "E:\\projects\\ML_FP\\flexpower_causal_transformer\\data\\processed\\test_data.pkl"

	# 加载pkl文件
	data = load_pkl_file(pkl_file_path)

	# 可视化数据
	if data is not None:
		visualize_data(data)
