"""
Flex Power Detection Results Visualization (Python Version)
Replacement for MATLAB visualization script with identical functionality
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class FlexPowerMatlabVisualizer:
	"""Python replacement for MATLAB visualization script"""

	def __init__(self, results_dir: str = 'test_results'):
		"""
		Initialize visualizer with test results directory

		Args:
			results_dir: Directory containing test results
		"""
		self.results_dir = results_dir
		self.data = None
		self.raw_data = None
		self.metadata = None

	def load_results(self, results_file: str = 'test_results.pkl') -> bool:
		"""
		Load test results from pickle file (mimics MATLAB's data loading)

		Args:
			results_file: Name of the results file

		Returns:
			bool: True if successfully loaded
		"""
		# Find the most recent results directory
		if os.path.exists(self.results_dir):
			dirs = [d for d in os.listdir(self.results_dir) if d.startswith('20')]
			if dirs:
				dirs.sort()
				latest_dir = os.path.join(self.results_dir, dirs[-1])
				pkl_file = os.path.join(latest_dir, results_file)
			else:
				print(f"No results directories found in {self.results_dir}")
				return False
		else:
			pkl_file = results_file

		print(f"Loading data from: {pkl_file}")

		try:
			with open(pkl_file, 'rb') as f:
				self.data = pickle.load(f)
			print("Data loaded successfully!")

			# Extract main arrays
			self.predictions = self.data['predictions']
			self.labels = self.data['labels']
			self.probabilities = self.data['probabilities']
			self.raw_data = self.data['raw_data']
			self.metadata = self.data.get('metadata', [])

			return True

		except Exception as e:
			print(f"Error loading pickle file: {e}")
			return False

	def extract_cnr_data(self):
		"""Extract CNR data from raw data (mimics MATLAB data extraction)"""
		n = len(self.raw_data)

		self.s2w_current = np.zeros(n)
		self.s1c_current = np.zeros(n)
		self.diff_current = np.zeros(n)
		self.satellite_prn = []
		self.elevation = np.zeros(n)

		for i, sample in enumerate(self.raw_data):
			self.s2w_current[i] = sample['s2w_current']
			self.s1c_current[i] = sample['s1c_current']
			self.diff_current[i] = sample['diff_current']
			self.satellite_prn.append(sample['satellite_prn'])

			if 'elevation' in sample and sample['elevation'] is not None:
				self.elevation[i] = sample['elevation']
			else:
				self.elevation[i] = np.nan

	def find_consecutive_regions(self, binary_vector):
		"""
		Find start and end indices of consecutive 1s in a binary vector
		(Direct translation of MATLAB helper function)
		"""
		regions = []
		in_region = False
		start_idx = 0

		for i in range(len(binary_vector)):
			if binary_vector[i] and not in_region:
				in_region = True
				start_idx = i
			elif not binary_vector[i] and in_region:
				in_region = False
				regions.append([start_idx, i - 1])

		# Handle case where region extends to the end
		if in_region:
			regions.append([start_idx, len(binary_vector) - 1])

		return regions

	def plot_time_series_matlab_style(self, satellite_prn: str = 'G01',
									  max_points: int = 500,
									  save_path: Optional[str] = None):
		"""
		Create time series visualization identical to MATLAB version

		Args:
			satellite_prn: Satellite PRN to visualize
			max_points: Maximum number of points to display
			save_path: Path to save the figure
		"""
		# Extract data for selected satellite
		sat_indices = [i for i, prn in enumerate(self.satellite_prn)
					   if prn == satellite_prn]

		if not sat_indices:
			print(f"No data found for satellite {satellite_prn}")
			return

		# Limit number of points
		if len(sat_indices) > max_points:
			sat_indices = sat_indices[:max_points]

		# Extract satellite-specific data
		sat_s2w = self.s2w_current[sat_indices]
		sat_s1c = self.s1c_current[sat_indices]
		sat_diff = self.diff_current[sat_indices]
		sat_labels = self.labels[sat_indices]
		sat_predictions = self.predictions[sat_indices]
		sat_probs = self.probabilities[sat_indices]
		sat_elevation = self.elevation[sat_indices]

		time_axis = np.arange(len(sat_s2w))

		# Create main figure (matching MATLAB layout)
		fig = plt.figure(figsize=(14, 8))
		fig.suptitle(f'Flex Power Detection Results - Satellite {satellite_prn}',
					 fontsize=14, fontweight='bold')

		# --- Subplot 1: S2W CNR Time Series ---
		ax1 = plt.subplot(3, 1, 1)

		# Plot base time series
		ax1.plot(time_axis, sat_s2w, 'b-', linewidth=0.5, alpha=0.3)

		# Add shaded regions for true Flex Power periods
		flex_regions = self.find_consecutive_regions(sat_labels == 1)
		for region in flex_regions:
			ax1.axvspan(time_axis[region[0]], time_axis[region[1]],
						alpha=0.1, color='orange')

		# Overlay colored points based on truth and predictions
		for i in range(len(sat_s2w)):
			if sat_labels[i] == 1 and sat_predictions[i] == 1:
				# Both true and predicted as Flex Power (purple)
				ax1.plot(time_axis[i], sat_s2w[i], 'o', color=[0.5, 0, 0.5],
						 markerfacecolor=[0.5, 0, 0.5], markersize=6)
			elif sat_labels[i] == 1:
				# True Flex Power (orange)
				ax1.plot(time_axis[i], sat_s2w[i], '^', color=[1, 0.5, 0],
						 markerfacecolor=[1, 0.5, 0], markersize=6)
			elif sat_predictions[i] == 1:
				# Predicted Flex Power (red)
				ax1.plot(time_axis[i], sat_s2w[i], 'v', color='r',
						 markerfacecolor='r', markersize=6)
			else:
				# Normal (blue)
				ax1.plot(time_axis[i], sat_s2w[i], 'o', color='b',
						 markerfacecolor='b', markersize=3, markeredgecolor='none')

		ax1.set_ylabel('S2W CNR (dB-Hz)', fontsize=11)
		ax1.set_title(f'S2W Signal with Flex Power Detection', fontsize=12)
		ax1.grid(True, alpha=0.3)

		# Create legend
		legend_elements = [
			mpatches.Patch(color=[1, 0.5, 0], label='Truth: ON'),
			mpatches.Patch(color='red', label='Predicted: ON'),
			mpatches.Patch(color='blue', label='Normal'),
			mpatches.Patch(color=[0.5, 0, 0.5], label='Both ON')
		]
		ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

		# --- Subplot 2: S1C CNR Time Series ---
		ax2 = plt.subplot(3, 1, 2)
		ax2.plot(time_axis, sat_s1c, 'g-', linewidth=1, label='S1C')
		ax2.set_ylabel('S1C CNR (dB-Hz)', fontsize=11)
		ax2.set_title('S1C Reference Signal', fontsize=12)
		ax2.grid(True, alpha=0.3)
		ax2.legend(loc='upper right')

		# --- Subplot 3: Differential CNR ---
		ax3 = plt.subplot(3, 1, 3)

		# Plot base differential
		ax3.plot(time_axis, sat_diff, 'k-', linewidth=0.5, alpha=0.3)

		# Overlay colored points
		for i in range(len(sat_diff)):
			if sat_labels[i] == 1 and sat_predictions[i] == 1:
				ax3.plot(time_axis[i], sat_diff[i], 'o', color=[0.5, 0, 0.5],
						 markerfacecolor=[0.5, 0, 0.5], markersize=4)
			elif sat_labels[i] == 1:
				ax3.plot(time_axis[i], sat_diff[i], '^', color=[1, 0.5, 0],
						 markerfacecolor=[1, 0.5, 0], markersize=4)
			elif sat_predictions[i] == 1:
				ax3.plot(time_axis[i], sat_diff[i], 'v', color='r',
						 markerfacecolor='r', markersize=4)
			else:
				ax3.plot(time_axis[i], sat_diff[i], 'o', color='b',
						 markerfacecolor='b', markersize=2, markeredgecolor='none')

		# Reference line at zero
		ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

		ax3.set_xlabel('Time Index', fontsize=11)
		ax3.set_ylabel('S2W - S1C (dB)', fontsize=11)
		ax3.set_title('Differential CNR', fontsize=12)
		ax3.grid(True, alpha=0.3)

		# Calculate and display performance statistics
		correct = np.sum(sat_predictions == sat_labels)
		accuracy = correct / len(sat_labels)

		TP = np.sum((sat_predictions == 1) & (sat_labels == 1))
		FP = np.sum((sat_predictions == 1) & (sat_labels == 0))
		FN = np.sum((sat_predictions == 0) & (sat_labels == 1))
		TN = np.sum((sat_predictions == 0) & (sat_labels == 0))

		precision = TP / (TP + FP) if (TP + FP) > 0 else 0
		recall = TP / (TP + FN) if (TP + FN) > 0 else 0
		f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

		# Add statistics text box (mimicking MATLAB annotation)
		stats_text = (f'Statistics for {satellite_prn}:\n'
					  f'Accuracy: {accuracy * 100:.2f}%\n'
					  f'Precision: {precision * 100:.2f}%\n'
					  f'Recall: {recall * 100:.2f}%\n'
					  f'F1 Score: {f1_score * 100:.2f}%\n'
					  f'TP={TP}, FP={FP}, FN={FN}, TN={TN}')

		# Place text box in figure
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		fig.text(0.85, 0.7, stats_text, fontsize=10,
				 bbox=props, verticalalignment='top')

		plt.tight_layout(rect=[0, 0, 0.84, 0.96])

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"Figure saved to {save_path}")
		else:
			plt.show()

	def plot_confidence_analysis(self, satellite_prn: str = 'G01',
								 save_path: Optional[str] = None):
		"""
		Create confidence distribution analysis (matching MATLAB Figure 2)

		Args:
			satellite_prn: Satellite PRN to analyze
			save_path: Path to save the figure
		"""
		# Extract data for selected satellite
		sat_indices = [i for i, prn in enumerate(self.satellite_prn)
					   if prn == satellite_prn]

		if not sat_indices:
			print(f"No data found for satellite {satellite_prn}")
			return

		sat_labels = self.labels[sat_indices]
		sat_probs = self.probabilities[sat_indices, 1]  # Probability of Flex Power ON

		# Separate probabilities by true label
		pos_probs = sat_probs[sat_labels == 1]
		neg_probs = sat_probs[sat_labels == 0]

		# Create figure with two subplots
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
		fig.suptitle(f'Prediction Confidence Analysis - {satellite_prn}',
					 fontsize=14, fontweight='bold')

		# Subplot 1: Histogram of prediction probabilities
		ax1.hist(neg_probs, bins=20, alpha=0.5, color='green',
				 label='No Flex Power', edgecolor='black')
		ax1.hist(pos_probs, bins=20, alpha=0.5, color='orange',
				 label='Flex Power', edgecolor='black')
		ax1.set_xlabel('Predicted Probability of Flex Power', fontsize=11)
		ax1.set_ylabel('Count', fontsize=11)
		ax1.set_title('Prediction Confidence Distribution', fontsize=12)
		ax1.legend(loc='best', fontsize=10)
		ax1.grid(True, alpha=0.3)

		# Subplot 2: Box plot of probabilities
		bp = ax2.boxplot([neg_probs, pos_probs],
						 labels=['No Flex Power', 'Flex Power'],
						 patch_artist=True)

		# Color the boxes
		colors = ['green', 'orange']
		for patch, color in zip(bp['boxes'], colors):
			patch.set_facecolor(color)
			patch.set_alpha(0.5)

		ax2.set_ylabel('Predicted Probability', fontsize=11)
		ax2.set_title('Confidence by True Label', fontsize=12)
		ax2.grid(True, alpha=0.3)

		plt.tight_layout()

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"Figure saved to {save_path}")
		else:
			plt.show()

	def print_available_satellites(self):
		"""Print list of available satellites in the dataset"""
		unique_sats = list(set(self.satellite_prn))
		unique_sats.sort()

		print("\nAvailable satellites:")
		for i, sat in enumerate(unique_sats, 1):
			count = self.satellite_prn.count(sat)
			print(f"{i}. {sat} ({count} samples)")

	def run_complete_visualization(self, satellite_prn: str = 'G01',
								   output_dir: Optional[str] = None):
		"""
		Run complete visualization pipeline (matching MATLAB script)

		Args:
			satellite_prn: Satellite to visualize
			output_dir: Directory to save outputs
		"""
		# Load data
		if not self.load_results():
			return

		# Extract CNR data
		self.extract_cnr_data()

		# Print available satellites
		self.print_available_satellites()

		# Create output directory if specified
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)

		# Generate main time series plot
		print(f"\nGenerating time series plot for {satellite_prn}...")
		ts_path = os.path.join(output_dir, f'matlab_timeseries_{satellite_prn}.png') if output_dir else None
		self.plot_time_series_matlab_style(satellite_prn, save_path=ts_path)

		# Generate confidence analysis plot
		print(f"Generating confidence analysis for {satellite_prn}...")
		conf_path = os.path.join(output_dir, f'matlab_confidence_{satellite_prn}.png') if output_dir else None
		self.plot_confidence_analysis(satellite_prn, save_path=conf_path)

		if output_dir:
			print(f"\nAll figures saved to: {output_dir}")

		print("\nVisualization complete!")


def main():
	"""Main function to run MATLAB-style visualization"""
	import argparse

	parser = argparse.ArgumentParser(description='Flex Power Detection Visualization (MATLAB Style)')
	parser.add_argument('--results_dir', type=str, default='test_results',
						help='Directory containing test results')
	parser.add_argument('--satellite', type=str, default='G15',
						help='Satellite PRN to visualize (e.g., G01)')
	parser.add_argument('--output_dir', type=str, default='visualization_results/matlab_style',
						help='Directory to save output figures')
	parser.add_argument('--max_points', type=int, default=500,
						help='Maximum number of points to display')

	args = parser.parse_args()

	# Create visualizer
	visualizer = FlexPowerMatlabVisualizer(args.results_dir)

	# Run complete visualization
	visualizer.run_complete_visualization(
		satellite_prn=args.satellite,
		output_dir=args.output_dir
	)


if __name__ == "__main__":
	main()