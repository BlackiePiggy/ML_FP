import pickle
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np


def explore_pkl_structure(obj, indent=0, max_depth=3):
	"""递归打印对象的基本结构"""
	prefix = "  " * indent
	if isinstance(obj, dict):
		print(f"{prefix}字典 (keys={len(obj)}): {list(obj.keys())[:5]}...")
		if indent < max_depth:
			for k, v in list(obj.items())[:3]:  # 只看前三个
				print(f"{prefix}  Key={k}:")
				explore_pkl_structure(v, indent + 2, max_depth)
	elif isinstance(obj, (list, tuple)):
		print(f"{prefix}{type(obj).__name__} (len={len(obj)}), 示例: {obj[:3]}")
		if len(obj) > 0 and indent < max_depth:
			explore_pkl_structure(obj[0], indent + 1, max_depth)
	elif isinstance(obj, np.ndarray):
		print(f"{prefix}Numpy数组 shape={obj.shape}, dtype={obj.dtype}")
	elif isinstance(obj, pd.DataFrame):
		print(f"{prefix}DataFrame 形状={obj.shape}, 列={list(obj.columns)[:5]}...")
		print(obj.head(3))  # 打印前3行
	else:
		print(f"{prefix}{type(obj).__name__}: {str(obj)[:100]}")


def main():
	# 弹出文件选择框，默认路径
	root = tk.Tk()
	root.withdraw()  # 隐藏主窗口
	file_path = filedialog.askopenfilename(
		title="选择一个PKL文件",
		filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
		initialdir=r"E:\projects\ML_FP\data\4_raw_dataset"  # 默认路径
	)

	if not file_path:
		print("未选择文件")
		return

	print(f"加载文件: {file_path}")

	# 读取pkl
	with open(file_path, "rb") as f:
		data = pickle.load(f)

	print("\n==== 文件内容结构 ====")
	explore_pkl_structure(data)


if __name__ == "__main__":
	main()
