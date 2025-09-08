classdef SatelliteDataLabeler < matlab.apps.AppBase
    % 卫星信号数据标注工具
    % 用于批量标注卫星信号时序数据
    
    properties (Access = public)
        UIFigure                matlab.ui.Figure
        GridLayout              matlab.ui.container.GridLayout
        
        % 左侧面板控件
        LeftPanel               matlab.ui.container.Panel
        LoadButton              matlab.ui.control.Button
        StationListBox          matlab.ui.control.ListBox
        StationLabel            matlab.ui.control.Label
        SatelliteListBox        matlab.ui.control.ListBox
        SatelliteLabel          matlab.ui.control.Label
        DOYListBox              matlab.ui.control.ListBox
        DOYLabel                matlab.ui.control.Label
        PlotButton              matlab.ui.control.Button
        
        % 标注控件
        LabelPanel              matlab.ui.container.Panel
        StartPointButton        matlab.ui.control.Button
        EndPointButton          matlab.ui.control.Button
        Label1Button            matlab.ui.control.Button
        Label0Button            matlab.ui.control.Button
        ClearLabelsButton       matlab.ui.control.Button
        SaveButton              matlab.ui.control.Button
        OutputPathButton        matlab.ui.control.Button
        OutputPathField         matlab.ui.control.EditField
        
        % 右侧绘图区
        UIAxes                  matlab.ui.control.UIAxes
        
        % 状态显示
        StatusLabel             matlab.ui.control.Label
    end
    
    properties (Access = private)
        LoadedData              % 存储加载的数据
        PlottedData             % 存储绘制的数据
        StartPoint              % 标注起始点
        EndPoint                % 标注结束点
        Labels                  % 存储标注信息
        DataCursor              % 数据游标
        OutputPath              % 输出路径
        CurrentSelection        % 当前选中的数据点
    end
    
    methods (Access = private)
        
        function startupFcn(app)
            % 初始化
            app.LoadedData = struct();
            app.PlottedData = [];
            app.Labels = struct();
            app.OutputPath = '3_label_raw_datasets';
            app.OutputPathField.Value = app.OutputPath;
            
            % 设置列表框为多选模式
            app.StationListBox.Multiselect = 'on';
            app.SatelliteListBox.Multiselect = 'on';
            app.DOYListBox.Multiselect = 'on';
            
            % 初始化状态
            app.StatusLabel.Text = '请加载数据文件';
        end
        
        function LoadButtonPushed(app, event)
            % 加载CSV文件
            [files, path] = uigetfile('*.csv', '选择CSV文件', ...
                '2_raw_datasets', 'MultiSelect', 'on');
            
            if isequal(files, 0)
                return;
            end
            
            if ~iscell(files)
                files = {files};
            end
            
            app.StatusLabel.Text = '正在加载文件...';
            drawnow;
            
            stations = {};
            satellites = {};
            doys = {};
            
            for i = 1:length(files)
                filename = files{i};
                filepath = fullfile(path, filename);
                
                % 解析文件名
                [~, name, ~] = fileparts(filename);
                parts = split(name, '_');
                if length(parts) >= 3
                    station = parts{1};
                    satellite = parts{2};
                    doy = parts{3};
                    
                    % 读取CSV文件
                    try
                        data = readtable(filepath);
                        
                        % 存储数据
                        key = sprintf('%s_%s_%s', station, satellite, doy);
                        app.LoadedData.(key) = data;
                        
                        % 收集唯一值
                        if ~ismember(station, stations)
                            stations{end+1} = station;
                        end
                        if ~ismember(satellite, satellites)
                            satellites{end+1} = satellite;
                        end
                        if ~ismember(doy, doys)
                            doys{end+1} = doy;
                        end
                        
                        % 初始化标签列
                        if ~ismember('label', data.Properties.VariableNames)
                            app.LoadedData.(key).label = zeros(height(data), 1);
                        end
                        
                    catch ME
                        fprintf('加载文件 %s 失败: %s\n', filename, ME.message);
                    end
                end
            end
            
            % 更新列表框
            app.StationListBox.Items = stations;
            app.SatelliteListBox.Items = satellites;
            app.DOYListBox.Items = doys;
            
            app.StatusLabel.Text = sprintf('已加载 %d 个文件', length(files));
        end
        
        function PlotButtonPushed(app, event)
            % 绘制选中的时间序列
            stations = app.StationListBox.Value;
            satellites = app.SatelliteListBox.Value;
            doys = app.DOYListBox.Value;
            
            if isempty(stations) || isempty(satellites) || isempty(doys)
                app.StatusLabel.Text = '请选择要绘制的测站、卫星和DOY';
                return;
            end
            
            cla(app.UIAxes);
            hold(app.UIAxes, 'on');
            
            colors = lines(length(stations) * length(satellites));
            colorIdx = 1;
            app.PlottedData = [];
            legendEntries = {};
            
            for i = 1:length(stations)
                for j = 1:length(satellites)
                    xData = [];
                    yData = [];
                    indices = [];
                    keys = {};
                    
                    for k = 1:length(doys)
                        key = sprintf('%s_%s_%s', stations{i}, satellites{j}, doys{k});
                        
                        if isfield(app.LoadedData, key)
                            data = app.LoadedData.(key);
                            
                            % 转换时间并合并数据
                            timeData = datetime(data.time_utc, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
                            xData = [xData; timeData];
                            yData = [yData; data.S2W_S1C_diff];
                            
                            % 记录数据索引和键值
                            n = height(data);
                            indices = [indices; (1:n)'];
                            keys = [keys; repmat({key}, n, 1)];
                        end
                    end
                    
                    if ~isempty(xData)
                        % 绘制散点图
                        scatter(app.UIAxes, xData, yData, 20, colors(colorIdx, :), 'filled');
                        
                        % 存储绘制的数据信息
                        plotInfo = struct();
                        plotInfo.station = stations{i};
                        plotInfo.satellite = satellites{j};
                        plotInfo.xData = xData;
                        plotInfo.yData = yData;
                        plotInfo.indices = indices;
                        plotInfo.keys = keys;
                        plotInfo.color = colors(colorIdx, :);
                        
                        app.PlottedData = [app.PlottedData, plotInfo];
                        
                        legendEntries{end+1} = sprintf('%s_%s', stations{i}, satellites{j});
                        colorIdx = colorIdx + 1;
                    end
                end
            end
            
            hold(app.UIAxes, 'off');
            xlabel(app.UIAxes, '时间 (UTC)');
            ylabel(app.UIAxes, 'S2W-S1C 差值');
            title(app.UIAxes, '卫星信号时序数据');
            legend(app.UIAxes, legendEntries, 'Location', 'best');
            grid(app.UIAxes, 'on');
            
            % 启用数据游标
            dcm = datacursormode(app.UIFigure);
            dcm.Enable = 'on';
            dcm.UpdateFcn = @(src, evt) app.dataCursorCallback(src, evt);
            
            app.StatusLabel.Text = '绘图完成，可以使用数据游标选择点';
        end
        
        function txt = dataCursorCallback(app, ~, evt)
            % 数据游标回调函数
            pos = evt.Position;
            
            % 找到最近的数据点
            minDist = inf;
            for i = 1:length(app.PlottedData)
                xNum = datenum(app.PlottedData(i).xData);
                dist = sqrt((xNum - datenum(pos(1))).^2 + ...
                           (app.PlottedData(i).yData - pos(2)).^2);
                [minD, idx] = min(dist);
                
                if minD < minDist
                    minDist = minD;
                    app.CurrentSelection = struct();
                    app.CurrentSelection.plotIdx = i;
                    app.CurrentSelection.dataIdx = idx;
                    app.CurrentSelection.key = app.PlottedData(i).keys{idx};
                    app.CurrentSelection.index = app.PlottedData(i).indices(idx);
                    app.CurrentSelection.time = app.PlottedData(i).xData(idx);
                    app.CurrentSelection.value = app.PlottedData(i).yData(idx);
                end
            end
            
            % 显示信息
            txt = {sprintf('时间: %s', datestr(app.CurrentSelection.time)), ...
                   sprintf('值: %.2f', app.CurrentSelection.value), ...
                   sprintf('文件: %s', app.CurrentSelection.key), ...
                   sprintf('索引: %d', app.CurrentSelection.index)};
        end
        
        function StartPointButtonPushed(app, event)
            % 设置起始点
            if isempty(app.CurrentSelection)
                app.StatusLabel.Text = '请先选择一个数据点';
                return;
            end
            
            app.StartPoint = app.CurrentSelection;
            app.StatusLabel.Text = sprintf('起始点已设置: %s', ...
                datestr(app.StartPoint.time));
        end
        
        function EndPointButtonPushed(app, event)
            % 设置结束点
            if isempty(app.CurrentSelection)
                app.StatusLabel.Text = '请先选择一个数据点';
                return;
            end
            
            app.EndPoint = app.CurrentSelection;
            app.StatusLabel.Text = sprintf('结束点已设置: %s', ...
                datestr(app.EndPoint.time));
        end
        
        function Label1ButtonPushed(app, event)
            % 标注为1
            app.labelRange(1);
        end
        
        function Label0ButtonPushed(app, event)
            % 标注为0
            app.labelRange(0);
        end
        
        function labelRange(app, labelValue)
            % 对选中范围进行标注
            if isempty(app.StartPoint) || isempty(app.EndPoint)
                app.StatusLabel.Text = '请先设置起始点和结束点';
                return;
            end
            
            % 确定时间范围
            startTime = app.StartPoint.time;
            endTime = app.EndPoint.time;
            if startTime > endTime
                temp = startTime;
                startTime = endTime;
                endTime = temp;
            end
            
            % 对所有绘制的数据进行标注
            labelCount = 0;
            for i = 1:length(app.PlottedData)
                timeInRange = app.PlottedData(i).xData >= startTime & ...
                             app.PlottedData(i).xData <= endTime;
                
                for j = find(timeInRange)'
                    key = app.PlottedData(i).keys{j};
                    idx = app.PlottedData(i).indices(j);
                    
                    if isfield(app.LoadedData, key)
                        app.LoadedData.(key).label(idx) = labelValue;
                        labelCount = labelCount + 1;
                    end
                end
            end
            
            app.StatusLabel.Text = sprintf('已标注 %d 个点为 %d', labelCount, labelValue);
            
            % 清除起始点和结束点
            app.StartPoint = [];
            app.EndPoint = [];
        end
        
        function ClearLabelsButtonPushed(app, event)
            % 清除所有标注
            fields = fieldnames(app.LoadedData);
            for i = 1:length(fields)
                app.LoadedData.(fields{i}).label(:) = 0;
            end
            app.StatusLabel.Text = '已清除所有标注';
        end
        
        function OutputPathButtonPushed(app, event)
            % 选择输出路径
            folder = uigetdir(app.OutputPath, '选择输出文件夹');
            if folder ~= 0
                app.OutputPath = folder;
                app.OutputPathField.Value = folder;
            end
        end
        
        function SaveButtonPushed(app, event)
            % 保存标注结果
            if ~exist(app.OutputPath, 'dir')
                mkdir(app.OutputPath);
            end
            
            app.StatusLabel.Text = '正在保存文件...';
            drawnow;
            
            savedCount = 0;
            fields = fieldnames(app.LoadedData);
            
            for i = 1:length(fields)
                key = fields{i};
                data = app.LoadedData.(key);
                
                % 确保有label列
                if ~ismember('label', data.Properties.VariableNames)
                    data.label = zeros(height(data), 1);
                end
                
                % 构建输出文件名
                filename = sprintf('%s.csv', strrep(key, '_', '_'));
                filepath = fullfile(app.OutputPath, filename);
                
                try
                    writetable(data, filepath);
                    savedCount = savedCount + 1;
                catch ME
                    fprintf('保存文件 %s 失败: %s\n', filename, ME.message);
                end
            end
            
            app.StatusLabel.Text = sprintf('已保存 %d 个文件到 %s', ...
                savedCount, app.OutputPath);
        end
    end
    
    % Component initialization
    methods (Access = private)
        
        function createComponents(app)
            % 创建 UIFigure 和组件
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1200 700];
            app.UIFigure.Name = '卫星信号数据标注工具';
            
            % 创建 GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {300, '1x'};
            app.GridLayout.RowHeight = {'1x', 30};
            
            % 创建左侧面板
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.Title = '控制面板';
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;
            
            % 左侧面板布局
            leftGrid = uigridlayout(app.LeftPanel);
            leftGrid.RowHeight = {30, 25, 80, 25, 80, 25, 80, 30, 180, 30};
            leftGrid.ColumnWidth = {'1x'};
            
            % 加载按钮
            app.LoadButton = uibutton(leftGrid, 'push');
            app.LoadButton.ButtonPushedFcn = createCallbackFcn(app, @LoadButtonPushed, true);
            app.LoadButton.Text = '加载CSV文件';
            app.LoadButton.Layout.Row = 1;
            app.LoadButton.Layout.Column = 1;
            
            % 测站列表
            app.StationLabel = uilabel(leftGrid);
            app.StationLabel.Text = '测站:';
            app.StationLabel.Layout.Row = 2;
            app.StationLabel.Layout.Column = 1;
            
            app.StationListBox = uilistbox(leftGrid);
            app.StationListBox.Layout.Row = 3;
            app.StationListBox.Layout.Column = 1;
            
            % 卫星列表
            app.SatelliteLabel = uilabel(leftGrid);
            app.SatelliteLabel.Text = '卫星:';
            app.SatelliteLabel.Layout.Row = 4;
            app.SatelliteLabel.Layout.Column = 1;
            
            app.SatelliteListBox = uilistbox(leftGrid);
            app.SatelliteListBox.Layout.Row = 5;
            app.SatelliteListBox.Layout.Column = 1;
            
            % DOY列表
            app.DOYLabel = uilabel(leftGrid);
            app.DOYLabel.Text = 'DOY:';
            app.DOYLabel.Layout.Row = 6;
            app.DOYLabel.Layout.Column = 1;
            
            app.DOYListBox = uilistbox(leftGrid);
            app.DOYListBox.Layout.Row = 7;
            app.DOYListBox.Layout.Column = 1;
            
            % 绘图按钮
            app.PlotButton = uibutton(leftGrid, 'push');
            app.PlotButton.ButtonPushedFcn = createCallbackFcn(app, @PlotButtonPushed, true);
            app.PlotButton.Text = '绘制时间序列';
            app.PlotButton.Layout.Row = 8;
            app.PlotButton.Layout.Column = 1;
            
            % 标注面板
            app.LabelPanel = uipanel(leftGrid);
            app.LabelPanel.Title = '标注控制';
            app.LabelPanel.Layout.Row = 9;
            app.LabelPanel.Layout.Column = 1;
            
            % 标注面板布局
            labelGrid = uigridlayout(app.LabelPanel);
            labelGrid.RowHeight = {30, 30, 30, 30, 30, 30};
            labelGrid.ColumnWidth = {'1x', '1x'};
            
            app.StartPointButton = uibutton(labelGrid, 'push');
            app.StartPointButton.ButtonPushedFcn = createCallbackFcn(app, @StartPointButtonPushed, true);
            app.StartPointButton.Text = '设置起始点';
            app.StartPointButton.Layout.Row = 1;
            app.StartPointButton.Layout.Column = 1;
            
            app.EndPointButton = uibutton(labelGrid, 'push');
            app.EndPointButton.ButtonPushedFcn = createCallbackFcn(app, @EndPointButtonPushed, true);
            app.EndPointButton.Text = '设置结束点';
            app.EndPointButton.Layout.Row = 1;
            app.EndPointButton.Layout.Column = 2;
            
            app.Label1Button = uibutton(labelGrid, 'push');
            app.Label1Button.ButtonPushedFcn = createCallbackFcn(app, @Label1ButtonPushed, true);
            app.Label1Button.Text = '标注为 1';
            app.Label1Button.Layout.Row = 2;
            app.Label1Button.Layout.Column = 1;
            
            app.Label0Button = uibutton(labelGrid, 'push');
            app.Label0Button.ButtonPushedFcn = createCallbackFcn(app, @Label0ButtonPushed, true);
            app.Label0Button.Text = '标注为 0';
            app.Label0Button.Layout.Row = 2;
            app.Label0Button.Layout.Column = 2;
            
            app.ClearLabelsButton = uibutton(labelGrid, 'push');
            app.ClearLabelsButton.ButtonPushedFcn = createCallbackFcn(app, @ClearLabelsButtonPushed, true);
            app.ClearLabelsButton.Text = '清除所有标注';
            app.ClearLabelsButton.Layout.Row = 3;
            app.ClearLabelsButton.Layout.Column = [1 2];
            
            app.OutputPathButton = uibutton(labelGrid, 'push');
            app.OutputPathButton.ButtonPushedFcn = createCallbackFcn(app, @OutputPathButtonPushed, true);
            app.OutputPathButton.Text = '选择输出路径';
            app.OutputPathButton.Layout.Row = 4;
            app.OutputPathButton.Layout.Column = [1 2];
            
            app.OutputPathField = uieditfield(labelGrid, 'text');
            app.OutputPathField.Layout.Row = 5;
            app.OutputPathField.Layout.Column = [1 2];
            
            app.SaveButton = uibutton(labelGrid, 'push');
            app.SaveButton.ButtonPushedFcn = createCallbackFcn(app, @SaveButtonPushed, true);
            app.SaveButton.Text = '保存标注结果';
            app.SaveButton.Layout.Row = 6;
            app.SaveButton.Layout.Column = [1 2];
            
            % 创建 UIAxes
            app.UIAxes = uiaxes(app.GridLayout);
            app.UIAxes.Layout.Row = 1;
            app.UIAxes.Layout.Column = 2;
            
            % 创建状态标签
            app.StatusLabel = uilabel(app.GridLayout);
            app.StatusLabel.Layout.Row = 2;
            app.StatusLabel.Layout.Column = [1 2];
            app.StatusLabel.Text = '准备就绪';
            
            % 显示 figure
            app.UIFigure.Visible = 'on';
        end
    end
    
    % App creation and deletion
    methods (Access = public)
        
        function app = SatelliteDataLabeler
            % 创建组件
            createComponents(app)
            
            % 注册应用程序
            registerApp(app, app.UIFigure)
            
            % 运行启动函数
            runStartupFcn(app, @startupFcn)
            
            if nargout == 0
                clear app
            end
        end
        
        function delete(app)
            % 删除应用程序
            delete(app.UIFigure)
        end
    end
end