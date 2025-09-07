%% Flex Power Detection Results Visualization
% This script loads the test results from Python and creates visualizations
% Author: Flex Power Detection System
% Date: 2025

clear; close all; clc;

%% Configuration
% Path to the test results file
results_dir = 'E:\projects\ML_FP\flexpower_causal_transformer\test_results\';  % Update this path as needed
results_file = 'test_results.pkl';  % Or 'matlab_data.pkl' for simplified version

% Find the most recent results directory
if exist(results_dir, 'dir')
    dirs = dir(fullfile(results_dir, '20*'));
    if ~isempty(dirs)
        [~, idx] = max([dirs.datenum]);
        latest_dir = fullfile(results_dir, dirs(idx).name);
        pkl_file = fullfile(latest_dir, results_file);
    else
        error('No results directories found in %s', results_dir);
    end
else
    % Direct file path
    pkl_file = results_file;
end

fprintf('Loading data from: %s\n', pkl_file);

%% Load Python pickle file
% Note: This requires Python to be installed and configured with MATLAB
try
    % Load the pickle file using Python
    data = py_load_pickle(pkl_file);
    fprintf('Data loaded successfully!\n');
catch ME
    fprintf('Error loading pickle file: %s\n', ME.message);
    fprintf('Trying alternative loading method...\n');

    % Alternative: Use system command to convert pickle to MAT file
    python_script = create_converter_script();
    system(sprintf('python %s %s', python_script, pkl_file));
    mat_file = strrep(pkl_file, '.pkl', '.mat');
    data = load(mat_file);
    delete(python_script);  % Clean up temporary script
end

%% Extract data
predictions = double(data.predictions);
labels = double(data.labels);
probabilities = double(data.probabilities);

% Extract raw CNR data
s2w_current = zeros(length(data.raw_data), 1);
s1c_current = zeros(length(data.raw_data), 1);
diff_current = zeros(length(data.raw_data), 1);
satellite_prn = cell(length(data.raw_data), 1);
elevation = zeros(length(data.raw_data), 1);

for i = 1:length(data.raw_data)
    s2w_current(i) = data.raw_data{i}.s2w_current;
    s1c_current(i) = data.raw_data{i}.s1c_current;
    diff_current(i) = data.raw_data{i}.diff_current;
    satellite_prn{i} = data.raw_data{i}.satellite_prn;

    if isfield(data.raw_data{i}, 'elevation') && ~isempty(data.raw_data{i}.elevation)
        elevation(i) = data.raw_data{i}.elevation;
    else
        elevation(i) = NaN;
    end
end

%% Select satellite for visualization
unique_sats = unique(satellite_prn);
fprintf('\nAvailable satellites:\n');
for i = 1:length(unique_sats)
    fprintf('%d. %s\n', i, unique_sats{i});
end

% Select first satellite by default, or specify
selected_sat = 'G01';  % Change this to select different satellite
sat_indices = strcmp(satellite_prn, selected_sat);

% Extract data for selected satellite
sat_s2w = s2w_current(sat_indices);
sat_s1c = s1c_current(sat_indices);
sat_diff = diff_current(sat_indices);
sat_labels = labels(sat_indices);
sat_predictions = predictions(sat_indices);
sat_probs = probabilities(sat_indices, :);
sat_elevation = elevation(sat_indices);

% Limit number of points for better visualization
max_points = 500;
if length(sat_s2w) > max_points
    indices = 1:max_points;
    sat_s2w = sat_s2w(indices);
    sat_s1c = sat_s1c(indices);
    sat_diff = sat_diff(indices);
    sat_labels = sat_labels(indices);
    sat_predictions = sat_predictions(indices);
    sat_probs = sat_probs(indices, :);
    sat_elevation = sat_elevation(indices);
end

time_axis = 1:length(sat_s2w);

%% Create Main Visualization
figure('Position', [100, 100, 1400, 800], 'Name', sprintf('Flex Power Detection - %s', selected_sat));

% --- Subplot 1: S2W CNR Time Series ---
subplot(3, 1, 1);
hold on;

% Plot base time series in blue
plot(time_axis, sat_s2w, 'b-', 'LineWidth', 0.5, 'DisplayName', 'S2W CNR');

% Overlay points with colors based on truth and predictions
for i = 1:length(sat_s2w)
    if sat_labels(i) == 1 && sat_predictions(i) == 1
        % Both true and predicted as Flex Power
        plot(time_axis(i), sat_s2w(i), 'o', 'Color', [0.5, 0, 0.5], ...
            'MarkerFaceColor', [0.5, 0, 0.5], 'MarkerSize', 6);
    elseif sat_labels(i) == 1
        % True Flex Power (orange)
        plot(time_axis(i), sat_s2w(i), '^', 'Color', [1, 0.5, 0], ...
            'MarkerFaceColor', [1, 0.5, 0], 'MarkerSize', 6);
    elseif sat_predictions(i) == 1
        % Predicted Flex Power (red)
        plot(time_axis(i), sat_s2w(i), 'v', 'Color', 'r', ...
            'MarkerFaceColor', 'r', 'MarkerSize', 6);
    else
        % Normal (blue)
        plot(time_axis(i), sat_s2w(i), 'o', 'Color', 'b', ...
            'MarkerFaceColor', 'b', 'MarkerSize', 3, 'MarkerEdgeColor', 'none');
    end
end

% Add shaded regions for true Flex Power periods
flex_regions = find_consecutive_regions(sat_labels == 1);
for i = 1:size(flex_regions, 1)
    x_start = time_axis(flex_regions(i, 1));
    x_end = time_axis(flex_regions(i, 2));
    y_lim = ylim;
    patch([x_start, x_end, x_end, x_start], ...
          [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
          [1, 0.5, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
end

grid on;
xlabel('Time Index');
ylabel('S2W CNR (dB-Hz)');
title(sprintf('Flex Power Detection Results - Satellite %s', selected_sat), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Create legend
h1 = plot(NaN, NaN, '^', 'Color', [1, 0.5, 0], 'MarkerFaceColor', [1, 0.5, 0], 'MarkerSize', 8);
h2 = plot(NaN, NaN, 'v', 'Color', 'r', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
h3 = plot(NaN, NaN, 'o', 'Color', 'b', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
h4 = plot(NaN, NaN, 'o', 'Color', [0.5, 0, 0.5], 'MarkerFaceColor', [0.5, 0, 0.5], 'MarkerSize', 8);
legend([h1, h2, h3, h4], {'Truth: ON', 'Predicted: ON', 'Normal', 'Both ON'}, ...
    'Location', 'best', 'FontSize', 10);

hold off;

% --- Subplot 2: S1C CNR Time Series ---
subplot(3, 1, 2);
plot(time_axis, sat_s1c, 'g-', 'LineWidth', 1);
grid on;
xlabel('Time Index');
ylabel('S1C CNR (dB-Hz)');
title('S1C Reference Signal', 'FontSize', 12);

% --- Subplot 3: Differential CNR ---
subplot(3, 1, 3);
hold on;

% Plot base differential
plot(time_axis, sat_diff, 'k-', 'LineWidth', 0.5);

% Overlay colored points
for i = 1:length(sat_diff)
    if sat_labels(i) == 1 && sat_predictions(i) == 1
        plot(time_axis(i), sat_diff(i), 'o', 'Color', [0.5, 0, 0.5], ...
            'MarkerFaceColor', [0.5, 0, 0.5], 'MarkerSize', 4);
    elseif sat_labels(i) == 1
        plot(time_axis(i), sat_diff(i), '^', 'Color', [1, 0.5, 0], ...
            'MarkerFaceColor', [1, 0.5, 0], 'MarkerSize', 4);
    elseif sat_predictions(i) == 1
        plot(time_axis(i), sat_diff(i), 'v', 'Color', 'r', ...
            'MarkerFaceColor', 'r', 'MarkerSize', 4);
    else
        plot(time_axis(i), sat_diff(i), 'o', 'Color', 'b', ...
            'MarkerFaceColor', 'b', 'MarkerSize', 2, 'MarkerEdgeColor', 'none');
    end
end

% Reference line at zero
plot(xlim, [0, 0], 'k--', 'LineWidth', 0.5, 'Alpha', 0.5);

grid on;
xlabel('Time Index');
ylabel('S2W - S1C (dB)');
title('Differential CNR', 'FontSize', 12);
hold off;

%% Performance Statistics
% Calculate metrics
correct_predictions = (sat_predictions == sat_labels);
accuracy = sum(correct_predictions) / length(correct_predictions);

TP = sum(sat_predictions == 1 & sat_labels == 1);
FP = sum(sat_predictions == 1 & sat_labels == 0);
FN = sum(sat_predictions == 0 & sat_labels == 1);
TN = sum(sat_predictions == 0 & sat_labels == 0);

precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * precision * recall / (precision + recall);

% Add text box with statistics
stats_text = sprintf(['Statistics for %s:\n' ...
    'Accuracy: %.2f%%\n' ...
    'Precision: %.2f%%\n' ...
    'Recall: %.2f%%\n' ...
    'F1 Score: %.2f%%\n' ...
    'TP=%d, FP=%d, FN=%d, TN=%d'], ...
    selected_sat, accuracy*100, precision*100, recall*100, f1_score*100, ...
    TP, FP, FN, TN);

annotation('textbox', [0.85, 0.7, 0.13, 0.2], ...
    'String', stats_text, ...
    'FontSize', 10, ...
    'BackgroundColor', 'w', ...
    'EdgeColor', 'k', ...
    'LineWidth', 1);

%% Additional Visualizations

% --- Figure 2: Confidence Distribution ---
figure('Position', [150, 150, 1200, 500], 'Name', 'Prediction Confidence Analysis');

subplot(1, 2, 1);
% Histogram of prediction probabilities
pos_probs = sat_probs(sat_labels == 1, 2);  % Probabilities for true positives
neg_probs = sat_probs(sat_labels == 0, 2);  % Probabilities for true negatives

histogram(neg_probs, 20, 'FaceColor', 'g', 'FaceAlpha', 0.5, 'DisplayName', 'No Flex Power');
hold on;
histogram(pos_probs, 20, 'FaceColor', [1, 0.5, 0], 'FaceAlpha', 0.5, 'DisplayName', 'Flex Power');
xlabel('Predicted Probability of Flex Power');
ylabel('Count');
title('Prediction Confidence Distribution');
legend('Location', 'best');
grid on;

subplot(1, 2, 2);
% Box plot of probabilities
boxplot([neg_probs; pos_probs], [zeros(size(neg_probs)); ones(size(pos_probs))], ...
    'Labels', {'No Flex Power', 'Flex Power'});
ylabel('Predicted Probability');
title('Confidence by True Label');
grid on;

%% Save figures
if exist('latest_dir', 'var')
    saveas(figure(1), fullfile(latest_dir, sprintf('matlab_timeseries_%s.png', selected_sat)));
    saveas(figure(2), fullfile(latest_dir, sprintf('matlab_confidence_%s.png', selected_sat)));
    fprintf('\nFigures saved to: %s\n', latest_dir);
end

fprintf('\nVisualization complete!\n');

%% Helper Functions

function regions = find_consecutive_regions(binary_vector)
    % Find start and end indices of consecutive 1s in a binary vector
    regions = [];
    in_region = false;
    start_idx = 0;

    for i = 1:length(binary_vector)
        if binary_vector(i) && ~in_region
            in_region = true;
            start_idx = i;
        elseif ~binary_vector(i) && in_region
            in_region = false;
            regions = [regions; start_idx, i-1];
        end
    end

    % Handle case where region extends to the end
    if in_region
        regions = [regions; start_idx, length(binary_vector)];
    end
end

function data = py_load_pickle(filename)
    % Load pickle file using Python interop
    py.importlib.import_module('pickle');
    fid = py.open(filename, 'rb');
    data = py.pickle.load(fid);
    fid.close();

    % Convert Python dict to MATLAB struct
    data = python_dict_to_matlab_struct(data);
end

function s = python_dict_to_matlab_struct(d)
    % Convert Python dictionary to MATLAB struct recursively
    if isa(d, 'py.dict')
        keys = cell(py.list(d.keys()));
        s = struct();
        for i = 1:length(keys)
            key = char(keys{i});
            value = d{keys{i}};

            % Clean field name for MATLAB
            clean_key = matlab.lang.makeValidName(key);

            if isa(value, 'py.dict')
                s.(clean_key) = python_dict_to_matlab_struct(value);
            elseif isa(value, 'py.list')
                s.(clean_key) = cell(value);
            elseif isa(value, 'py.numpy.ndarray')
                s.(clean_key) = double(value);
            else
                s.(clean_key) = value;
            end
        end
    else
        s = d;
    end
end

function script_name = create_converter_script()
    % Create a temporary Python script to convert pickle to MAT
    script_name = 'temp_converter.py';
    fid = fopen(script_name, 'w');
    fprintf(fid, 'import pickle\n');
    fprintf(fid, 'import scipy.io as sio\n');
    fprintf(fid, 'import sys\n');
    fprintf(fid, 'import numpy as np\n\n');
    fprintf(fid, 'def convert_to_matlab_compatible(obj):\n');
    fprintf(fid, '    if isinstance(obj, dict):\n');
    fprintf(fid, '        return {k: convert_to_matlab_compatible(v) for k, v in obj.items()}\n');
    fprintf(fid, '    elif isinstance(obj, list):\n');
    fprintf(fid, '        return [convert_to_matlab_compatible(item) for item in obj]\n');
    fprintf(fid, '    elif isinstance(obj, np.ndarray):\n');
    fprintf(fid, '        return obj\n');
    fprintf(fid, '    else:\n');
    fprintf(fid, '        return obj\n\n');
    fprintf(fid, 'with open(sys.argv[1], ''rb'') as f:\n');
    fprintf(fid, '    data = pickle.load(f)\n\n');
    fprintf(fid, 'matlab_data = convert_to_matlab_compatible(data)\n');
    fprintf(fid, 'output_file = sys.argv[1].replace(''.pkl'', ''.mat'')\n');
    fprintf(fid, 'sio.savemat(output_file, matlab_data)\n');
    fprintf(fid, 'print(f''Converted to {output_file}'')\n');
    fclose(fid);
end