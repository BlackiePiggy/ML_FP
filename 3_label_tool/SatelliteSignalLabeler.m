classdef SatelliteSignalLabeler < matlab.apps.AppBase
    % SatelliteSignalLabeler: 卫星S2W信号时序标注工具
    % - 加载CSV文件，基于S2W信号进行标注
    % - 右侧散点图：label=0 蓝色（正常）；label=1 红色（异常）
    % - 使用数据提示(DataTip)选点；拖动DataTip时实时更新选中点
    % - 设置"起始/结束时间"，对该单个区间一键标注为1或0
    % - 支持批量时间区间标注
    %
    % 运行：app = SatelliteSignalLabeler;

    properties (Access = public)
        UIFigure matlab.ui.Figure
        Grid     matlab.ui.container.GridLayout

        % 左侧面板控件
        LeftPanel          matlab.ui.container.Panel
        InputFolderLabel   matlab.ui.control.Label
        InputFolderEdit    matlab.ui.control.EditField
        PickInputBtn       matlab.ui.control.Button

        OutputFolderLabel  matlab.ui.control.Label
        OutputFolderEdit   matlab.ui.control.EditField
        PickOutputBtn      matlab.ui.control.Button
        
        LoadBtn            matlab.ui.control.Button
        SaveBtn            matlab.ui.control.Button

        FileLabel          matlab.ui.control.Label
        StartTimeLabel     matlab.ui.control.Label
        EndTimeLabel       matlab.ui.control.Label

        SelectedInfoLabel  matlab.ui.control.Label
        SetStartBtn        matlab.ui.control.Button
        SetEndBtn          matlab.ui.control.Button

        Mark1Btn           matlab.ui.control.Button
        Mark0Btn           matlab.ui.control.Button
        
        % 批量标注功能
        BatchMarkLabel     matlab.ui.control.Label
        BatchStartEdit     matlab.ui.control.EditField
        BatchEndEdit       matlab.ui.control.EditField
        BatchMark1Btn      matlab.ui.control.Button
        BatchMark0Btn      matlab.ui.control.Button

        % 右侧绘图
        RightPanel         matlab.ui.container.Panel
        Ax                 matlab.ui.control.UIAxes
    end

    properties (Access = private)
        % 原始数据表
        OriginalTable      table = table.empty
        
        % 核心数据
        T              datetime = datetime.empty
        Tnum           double   = []
        S2W            double   = []  % S2W信号值
        Label          double   = []  % 0/1

        % 绘图
        Sc             matlab.graphics.chart.primitive.Scatter = matlab.graphics.chart.primitive.Scatter.empty

        % 当前文件
        CurrentFile    string = ""
        CurrentFileName string = ""

        % 选择点与候选区间
        SelectedIdx    double = NaN
        StartCand      datetime = NaT
        EndCand        datetime = NaT

        % DataTip 监听器
        DTListener     event.listener = event.listener.empty

        % 拖动实时联动
        LastDTIndex    double = NaN
    end

    methods (Access = private)
        %% =============== UI 构建 ===============
        function buildUI(app)
            app.UIFigure = uifigure('Name','卫星S2W信号标注工具','Position',[100 100 1200 700]);
            app.UIFigure.Color = [0.98 0.98 0.99];

            % 鼠标移动时，若存在 DataTip，则实时同步选中点
            app.UIFigure.WindowButtonMotionFcn = @(~,~)app.onFigureMotion();

            app.Grid = uigridlayout(app.UIFigure,[1,2]);
            app.Grid.ColumnWidth = {380,'1x'};
            app.Grid.RowHeight = {'1x'};

            %% 左侧
            app.LeftPanel = uipanel(app.Grid,'Title','控制面板');
            app.LeftPanel.FontWeight = 'bold';
            gl = uigridlayout(app.LeftPanel,[13,3]);
            gl.RowHeight = repmat({28},1,13);
            gl.RowSpacing = 6;
            gl.ColumnWidth = {90,'1x',70};

            row = 1;
            
            % 输入文件夹（默认2_raw_datasets）
            app.InputFolderLabel = uilabel(gl,'Text','输入文件夹:','FontWeight','bold');
            app.InputFolderLabel.Layout.Row = row; app.InputFolderLabel.Layout.Column = 1;
            defaultInput = fullfile(pwd, '2_raw_datasets');
            if ~isfolder(defaultInput)
                defaultInput = pwd;
            end
            app.InputFolderEdit = uieditfield(gl,'text','Value',defaultInput);
            app.InputFolderEdit.Layout.Row = row; app.InputFolderEdit.Layout.Column = 2;
            app.PickInputBtn = uibutton(gl,'Text','选择','ButtonPushedFcn',@(~,~)app.pickInputFolder());
            app.PickInputBtn.Layout.Row = row; app.PickInputBtn.Layout.Column = 3;
            row = row + 1;

            % 输出文件夹（默认3_label_raw_datasets）
            app.OutputFolderLabel = uilabel(gl,'Text','输出文件夹:','FontWeight','bold');
            app.OutputFolderLabel.Layout.Row = row; app.OutputFolderLabel.Layout.Column = 1;
            defaultOutput = fullfile(pwd, '3_label_raw_datasets');
            if ~isfolder(defaultOutput)
                defaultOutput = pwd;
            end
            app.OutputFolderEdit = uieditfield(gl,'text','Value',defaultOutput);
            app.OutputFolderEdit.Layout.Row = row; app.OutputFolderEdit.Layout.Column = 2;
            app.PickOutputBtn = uibutton(gl,'Text','选择','ButtonPushedFcn',@(~,~)app.pickOutputFolder());
            app.PickOutputBtn.Layout.Row = row; app.PickOutputBtn.Layout.Column = 3;
            row = row + 1;

            % 载入/保存
            app.LoadBtn = uibutton(gl,'Text','载入CSV','ButtonPushedFcn',@(~,~)app.onLoadFile());
            app.LoadBtn.Layout.Row = row; app.LoadBtn.Layout.Column = [1 2];
            app.LoadBtn.BackgroundColor = [0.9 1 0.9];
            app.SaveBtn = uibutton(gl,'Text','保存标注','ButtonPushedFcn',@(~,~)app.onSave());
            app.SaveBtn.Layout.Row = row; app.SaveBtn.Layout.Column = 3;
            app.SaveBtn.BackgroundColor = [1 1 0.9];
            row = row + 1;

            % 文件&起止时间
            app.FileLabel = uilabel(gl,'Text','当前文件: -','Tooltip','显示已载入文件名');
            app.FileLabel.Layout.Row = row; app.FileLabel.Layout.Column = [1 3];
            row = row + 1;

            app.StartTimeLabel = uilabel(gl,'Text','数据起始: -');
            app.StartTimeLabel.Layout.Row = row; app.StartTimeLabel.Layout.Column = [1 3];
            row = row + 1;
            
            app.EndTimeLabel = uilabel(gl,'Text','数据结束: -');
            app.EndTimeLabel.Layout.Row = row; app.EndTimeLabel.Layout.Column = [1 3];
            row = row + 1;

            % 选点信息 & 起止设置
            app.SelectedInfoLabel = uilabel(gl,'Text','选中点: -','FontWeight','bold');
            app.SelectedInfoLabel.Layout.Row = row; app.SelectedInfoLabel.Layout.Column = [1 3];
            row = row + 1;

            % 设置起始/结束按钮
            subgrid = uigridlayout(gl,[1,2]);
            subgrid.Layout.Row = row;
            subgrid.Layout.Column = [1 3];
            subgrid.ColumnWidth = {'1x','1x'};
            subgrid.RowHeight   = {'1x'};
            subgrid.Padding     = [0 0 0 0];
            subgrid.ColumnSpacing = 6;

            app.SetStartBtn = uibutton(subgrid,'Text','设置为起始时间', ...
                'ButtonPushedFcn',@(~,~)app.onSetStart());
            app.SetStartBtn.Layout.Row = 1; app.SetStartBtn.Layout.Column = 1;

            app.SetEndBtn = uibutton(subgrid,'Text','设置为结束时间', ...
                'ButtonPushedFcn',@(~,~)app.onSetEnd());
            app.SetEndBtn.Layout.Row = 1; app.SetEndBtn.Layout.Column = 2;
            row = row + 1;

            app.Mark1Btn = uibutton(gl,'Text','标注为 1（异常）','ButtonPushedFcn',@(~,~)app.onMarkCurrent(1));
            app.Mark1Btn.Layout.Row = row; app.Mark1Btn.Layout.Column = [1 3];
            app.Mark1Btn.BackgroundColor = [1 0.8 0.8];
            row = row + 1;

            app.Mark0Btn = uibutton(gl,'Text','标注为 0（正常）','ButtonPushedFcn',@(~,~)app.onMarkCurrent(0));
            app.Mark0Btn.Layout.Row = row; app.Mark0Btn.Layout.Column = [1 3];
            app.Mark0Btn.BackgroundColor = [0.8 0.9 1];
            row = row + 1;
            
            % 批量标注区域
            app.BatchMarkLabel = uilabel(gl,'Text','批量标注时间区间:','FontWeight','bold');
            app.BatchMarkLabel.Layout.Row = row; app.BatchMarkLabel.Layout.Column = [1 3];
            row = row + 1;
            
            % 批量标注时间输入框和按钮
            batchGrid = uigridlayout(gl,[1,4]);
            batchGrid.Layout.Row = row;
            batchGrid.Layout.Column = [1 3];
            batchGrid.ColumnWidth = {'1x','1x',50,50};
            batchGrid.RowHeight   = {'1x'};
            batchGrid.Padding     = [0 0 0 0];
            batchGrid.ColumnSpacing = 4;
            
            app.BatchStartEdit = uieditfield(batchGrid,'text','Value','','Placeholder','HH:MM:SS');
            app.BatchStartEdit.Layout.Row = 1; app.BatchStartEdit.Layout.Column = 1;
            app.BatchStartEdit.Tooltip = '输入格式: HH:MM:SS 或 HH:MM';
            
            app.BatchEndEdit = uieditfield(batchGrid,'text','Value','','Placeholder','HH:MM:SS');
            app.BatchEndEdit.Layout.Row = 1; app.BatchEndEdit.Layout.Column = 2;
            app.BatchEndEdit.Tooltip = '输入格式: HH:MM:SS 或 HH:MM';
            
            app.BatchMark1Btn = uibutton(batchGrid,'Text','标1','ButtonPushedFcn',@(~,~)app.onBatchMark(1));
            app.BatchMark1Btn.Layout.Row = 1; app.BatchMark1Btn.Layout.Column = 3;
            app.BatchMark1Btn.BackgroundColor = [1 0.8 0.8];
            app.BatchMark1Btn.Tooltip = '将指定时间段标注为1（异常）';
            
            app.BatchMark0Btn = uibutton(batchGrid,'Text','标0','ButtonPushedFcn',@(~,~)app.onBatchMark(0));
            app.BatchMark0Btn.Layout.Row = 1; app.BatchMark0Btn.Layout.Column = 4;
            app.BatchMark0Btn.BackgroundColor = [0.8 0.9 1];
            app.BatchMark0Btn.Tooltip = '将指定时间段标注为0（正常）';

            %% 右侧绘图
            app.RightPanel = uipanel(app.Grid,'Title','S2W信号时序可视化');
            app.RightPanel.FontWeight = 'bold';
            app.Ax = uiaxes(app.RightPanel,'Position',[30 30 740 620]);

            try enableDefaultInteractivity(app.Ax); catch, end
            try app.Ax.Toolbar.Visible = 'on'; catch, end

            grid(app.Ax,'on');
            xlabel(app.Ax,'Time (UTC)');
            ylabel(app.Ax,'S2W Signal');
            title(app.Ax,'S2W Timeseries (蓝:正常, 红:异常)');
        end

        %% =============== 事件回调 ===============
        function pickInputFolder(app)
            p = uigetdir(app.InputFolderEdit.Value,'选择输入文件夹（2_raw_datasets）');
            if ischar(p) || (isstring(p) && strlength(p)>0)
                app.InputFolderEdit.Value = string(p);
            end
        end

        function pickOutputFolder(app)
            p = uigetdir(app.OutputFolderEdit.Value,'选择输出文件夹（3_label_raw_datasets）');
            if ischar(p) || (isstring(p) && strlength(p)>0)
                app.OutputFolderEdit.Value = string(p);
                % 如果文件夹不存在，创建它
                if ~isfolder(p)
                    mkdir(p);
                end
            end
        end

        function onLoadFile(app)
            startDir = app.InputFolderEdit.Value;
            if ~(isstring(startDir) || ischar(startDir)) || ~isfolder(startDir)
                startDir = pwd;
            end
            [f,p] = uigetfile({'*.csv','CSV Files (*.csv)'},'选择卫星信号CSV文件',startDir);
            if isequal(f,0); return; end
            app.CurrentFile = string(fullfile(p,f));
            app.CurrentFileName = string(f);
            app.FileLabel.Text = "当前文件: " + f;

            % 读表
            try
                app.OriginalTable = readtable(app.CurrentFile);
                
                % 检查必要的列是否存在
                if ~ismember('time_utc', app.OriginalTable.Properties.VariableNames)
                    uialert(app.UIFigure,'CSV文件缺少time_utc列','列缺失'); return;
                end
                if ~ismember('S2W', app.OriginalTable.Properties.VariableNames)
                    uialert(app.UIFigure,'CSV文件缺少S2W列','列缺失'); return;
                end
                
                % 解析时间列
                time_col = app.OriginalTable.time_utc;
                [t_dt, ok] = app.parseTimeColumn(time_col);
                if ~ok
                    uialert(app.UIFigure,'无法识别时间列格式','时间解析失败'); return;
                end
                
                app.T = t_dt(:);
                app.Tnum = datenum(app.T);
                app.S2W = double(app.OriginalTable.S2W(:));
                n = numel(app.S2W);
                
                % 检查是否已有label列
                if ismember('label', app.OriginalTable.Properties.VariableNames)
                    app.Label = double(app.OriginalTable.label(:));
                    uialert(app.UIFigure,'检测到已有label列，已加载现有标注','提示','Icon','info');
                else
                    app.Label = zeros(n,1);  % 默认全部为0
                end

                % 起止显示
                if ~isempty(app.T)
                    app.StartTimeLabel.Text = "数据起始: " + string(app.T(1));
                    app.EndTimeLabel.Text   = "数据结束: " + string(app.T(end));
                else
                    app.StartTimeLabel.Text = "数据起始: -";
                    app.EndTimeLabel.Text   = "数据结束: -";
                end

                % 绘图
                app.plotScatter();

                % 清空状态
                app.SelectedIdx = NaN;
                app.LastDTIndex = NaN;
                app.StartCand = NaT; app.EndCand = NaT;
                app.SelectedInfoLabel.Text = '选中点: -';

            catch ME
                uialert(app.UIFigure, sprintf('读取失败:\n%s', ME.message), '错误');
            end
        end

        function onSave(app)
            if isempty(app.T)
                uialert(app.UIFigure,'请先载入数据','未加载'); return;
            end
            
            % 在原始表格中添加或更新label列
            app.OriginalTable.label = app.Label;
            
            % 保存到输出文件夹，保持原文件名
            outFolder = string(app.OutputFolderEdit.Value);
            if ~isfolder(outFolder)
                mkdir(outFolder);
            end
            outFile = fullfile(outFolder, app.CurrentFileName);

            try
                writetable(app.OriginalTable, outFile);
                uialert(app.UIFigure, sprintf('已保存: %s\n标注统计: 0(正常)=%d个, 1(异常)=%d个', ...
                    outFile, sum(app.Label==0), sum(app.Label==1)), '保存成功', 'Icon', 'success');
            catch ME
                uialert(app.UIFigure, sprintf('保存失败:\n%s', ME.message), '错误');
            end
        end

        function onSetStart(app)
            if isnan(app.SelectedIdx)
                uialert(app.UIFigure,'请先选中一个点（点击散点或拖动数据提示）','未选中'); return;
            end
            app.StartCand = app.T(app.SelectedIdx);
            app.notifyCand();
        end

        function onSetEnd(app)
            if isnan(app.SelectedIdx)
                uialert(app.UIFigure,'请先选中一个点（点击散点或拖动数据提示）','未选中'); return;
            end
            app.EndCand = app.T(app.SelectedIdx);
            app.notifyCand();
        end

        function onMarkCurrent(app, val)
            % 使用当前候选起止时间对单个区间打标
            if isempty(app.T)
                uialert(app.UIFigure,'请先载入数据','未加载'); return;
            end
            if ismissing(app.StartCand) || ismissing(app.EndCand)
                uialert(app.UIFigure,'请先设置起始时间与结束时间','区间未设置'); return;
            end
            t1 = min(app.StartCand, app.EndCand);
            t2 = max(app.StartCand, app.EndCand);
            mask = (app.T >= t1) & (app.T <= t2);
            count = sum(mask);
            app.Label(mask) = val;
            app.refreshColors();
            
            % 显示标注结果
            labelText = sprintf('已将 %s 至 %s 的 %d 个点标注为 %d', ...
                datestr(t1,'HH:MM:SS'), datestr(t2,'HH:MM:SS'), count, val);
            app.SelectedInfoLabel.Text = labelText;
        end
        
        function onBatchMark(app, val)
            % 批量标注指定时间区间
            if isempty(app.T)
                uialert(app.UIFigure,'请先载入数据','未加载'); return;
            end
            
            startStr = app.BatchStartEdit.Value;
            endStr = app.BatchEndEdit.Value;
            
            if isempty(startStr) || isempty(endStr)
                uialert(app.UIFigure,'请输入起始和结束时间','时间未设置'); return;
            end
            
            try
                % 获取当前日期部分
                baseDate = dateshift(app.T(1),'start','day');
                
                % 解析时间字符串并组合日期
                startTime = app.parseTimeString(startStr, baseDate);
                endTime = app.parseTimeString(endStr, baseDate);
                
                if isnat(startTime) || isnat(endTime)
                    uialert(app.UIFigure,'时间格式错误，请使用 HH:MM:SS 或 HH:MM 格式','格式错误'); return;
                end
                
                % 执行标注
                mask = (app.T >= startTime) & (app.T <= endTime);
                count = sum(mask);
                
                if count == 0
                    uialert(app.UIFigure,'指定时间区间内没有数据点','无数据'); return;
                end
                
                app.Label(mask) = val;
                app.refreshColors();
                
                % 显示结果
                msg = sprintf('已将 %s 至 %s 的 %d 个点标注为 %d', ...
                    datestr(startTime,'HH:MM:SS'), datestr(endTime,'HH:MM:SS'), count, val);
                uialert(app.UIFigure, msg, '批量标注完成', 'Icon', 'success');
                
            catch ME
                uialert(app.UIFigure, sprintf('批量标注失败:\n%s', ME.message), '错误');
            end
        end

        %% =============== 绘图/颜色/数据提示 ===============
        function plotScatter(app)
            cla(app.Ax);
            if isempty(app.Tnum); return; end

            % 根据Label设置颜色
            n = numel(app.Tnum);
            c = zeros(n,3);
            blue = [0 0.447 0.741];  % 正常（0）
            red  = [0.85 0.1 0.1];   % 异常（1）
            for i=1:n
                c(i,:) = red .* (app.Label(i)==1) + blue .* (app.Label(i)==0);
            end
            
            app.Sc = scatter(app.Ax, app.Tnum, app.S2W, 12, c, 'filled');

            % X 轴显示为时间
            datetick(app.Ax,'x','keeplimits');
            title(app.Ax,'S2W Timeseries (蓝:正常, 红:异常)');
            xlabel(app.Ax,'Time (UTC)');
            ylabel(app.Ax,'S2W Signal');
            grid(app.Ax,'on');

            % 配置 DataTip 显示内容
            try
                row1 = dataTipTextRow('Time', string(app.T));
                row2 = dataTipTextRow('S2W', app.S2W);
                row3 = dataTipTextRow('Label', app.Label);
                app.Sc.DataTipTemplate.DataTipRows = [row1 row2 row3];
            catch
            end

            % 允许点击散点来创建/移动 DataTip
            app.Sc.PickableParts = 'all';
            app.Sc.HitTest = 'on';
            app.Sc.ButtonDownFcn = @(src,event)app.onScatterClick(event);

            % 监听 DataTip 创建事件
            try
                if ~isempty(app.DTListener) && isvalid(app.DTListener)
                    delete(app.DTListener);
                end
                app.DTListener = addlistener(app.Sc,'DataTipCreated',@(src,evt)app.onDataTipCreated(evt));
            catch
            end
        end

        function refreshColors(app)
            if isempty(app.Sc) || isempty(app.Label); return; end
            n = numel(app.Label);
            c = zeros(n,3);
            blue = [0 0.447 0.741];
            red  = [0.85 0.1 0.1];
            for i=1:n
                c(i,:) = red .* (app.Label(i)==1) + blue .* (app.Label(i)==0);
            end
            app.Sc.CData = c;

            % 同步 DataTip 的 Label 行
            try
                rows = app.Sc.DataTipTemplate.DataTipRows;
                hasLabelRow = false;
                for k = 1:numel(rows)
                    if strcmpi(rows(k).Label, 'Label')
                        rows(k) = dataTipTextRow('Label', app.Label);
                        hasLabelRow = true;
                        break;
                    end
                end
                if hasLabelRow
                    app.Sc.DataTipTemplate.DataTipRows = rows;
                else
                    row1 = dataTipTextRow('Time', string(app.T));
                    row2 = dataTipTextRow('S2W', app.S2W);
                    row3 = dataTipTextRow('Label', app.Label);
                    app.Sc.DataTipTemplate.DataTipRows = [row1 row2 row3];
                end
            catch
            end

            drawnow limitrate;
        end

        %% =============== DataTip 交互 ===============
        function onDataTipCreated(app, evt)
            try
                dt = evt.DataTip;
                idx = app.getIndexFromDataTip(dt);
                if ~isnan(idx)
                    app.setSelectedIdx(idx);
                    app.LastDTIndex = idx;
                    % 只保留一个 DataTip
                    dts = findall(app.Ax, 'Type', 'datatip');
                    if numel(dts) > 1
                        delete(dts(dts ~= dt));
                    end
                end
            catch
            end
        end

        function onFigureMotion(app)
            if isempty(app.Sc); return; end
            dt = app.getSingleDataTip();
            if isempty(dt); return; end

            idx = app.getIndexFromDataTip(dt);
            if isnan(idx); return; end

            if isnan(app.LastDTIndex) || idx ~= app.LastDTIndex
                app.setSelectedIdx(idx);
                app.LastDTIndex = idx;
            end
        end

        function onScatterClick(app, event)
            if isempty(app.Tnum); return; end

            x = []; y = [];
            try
                if isprop(event,'IntersectionPoint') && ~isempty(event.IntersectionPoint)
                    x = event.IntersectionPoint(1); y = event.IntersectionPoint(2);
                end
            catch
            end
            if isempty(x)
                curr = app.Ax.CurrentPoint;
                x = curr(1,1); y = curr(1,2);
            end

            X = app.Sc.XData; Y = app.Sc.YData;
            [~, idx] = min(abs(X - x) + abs(Y - y));

            % 创建或移动 DataTip
            try
                dts = findall(app.Ax,'Type','datatip');
                if isempty(dts)
                    datatip(app.Sc,'DataIndex',idx);
                else
                    try
                        dts(1).Target = app.Sc;
                        dts(1).DataIndex = idx;
                    catch
                        delete(dts);
                        datatip(app.Sc,'DataIndex',idx);
                    end
                end
            catch
            end

            app.setSelectedIdx(idx);
            app.LastDTIndex = idx;
        end

        function idx = getIndexFromDataTip(app, dt)
            idx = NaN;
            try
                if isprop(dt,'DataIndex')
                    di = dt.DataIndex;
                    if ~isempty(di) && isfinite(di)
                        idx = di;
                        return;
                    end
                end
            catch
            end
            % 用 Position 反算
            try
                pos = dt.Position;
                x = pos(1); y = pos(2);
                X = app.Sc.XData; Y = app.Sc.YData;
                [~, idx] = min(abs(X - x) + abs(Y - y));
            catch
                idx = NaN;
            end
        end

        function dt = getSingleDataTip(app)
            dt = [];
            try
                dts = findall(app.Ax,'Type','datatip');
                if isempty(dts); return; end
                dt = dts(1);
                if numel(dts) > 1
                    delete(dts(2:end));
                end
            catch
                dt = [];
            end
        end

        function setSelectedIdx(app, idx)
            if isempty(app.S2W) || idx<1 || idx>numel(app.S2W)
                return;
            end
            app.SelectedIdx = idx;
            app.SelectedInfoLabel.Text = sprintf('选中点: %s | S2W=%.2f | Label=%d', ...
                string(app.T(idx)), app.S2W(idx), app.Label(idx));
            app.flashPoint(idx);
        end

        function flashPoint(app, idx)
            if isempty(app.Sc) || idx<1 || idx>size(app.Sc.CData,1)
                return;
            end
            c = app.Sc.CData;
            base = c(idx,:);
            app.Sc.CData(idx,:) = [0 0 0];
            drawnow;
            pause(0.04);
            app.Sc.CData(idx,:) = base;
            drawnow;
        end

        function notifyCand(app)
            s = "(未设)";
            e = "(未设)";
            if ~ismissing(app.StartCand); s = string(app.StartCand); end
            if ~ismissing(app.EndCand);   e = string(app.EndCand);   end
            app.SelectedInfoLabel.Text = "候选区间: [" + s + "  ~  " + e + "]";
        end

        %% =============== 辅助函数 ===============
        function [t_dt, ok] = parseTimeColumn(~, tcol)
            ok = true; t_dt = datetime.empty;

            % 如果已经是datetime
            if isdatetime(tcol)
                t_dt = tcol; return;
            end

            % 如果是字符串或cell字符串
            if iscellstr(tcol) || isstring(tcol) || ischar(tcol)
                try
                    % 尝试解析，假设格式为 YYYY-MM-DD HH:MM:SS.FFF
                    t_dt = datetime(tcol, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS', 'TimeZone', 'UTC');
                    return;
                catch
                    try
                        % 尝试其他格式
                        t_dt = datetime(tcol, 'InputFormat', '', 'TimeZone', 'UTC');
                        return;
                    catch
                        ok = false;
                        return;
                    end
                end
            end

            % 如果是数值（可能是epoch时间戳）
            try
                num = double(tcol);
            catch
                ok = false; return;
            end

            m = median(num,'omitnan');

            % 判断时间戳类型
            if m > 1e11 && m < 2e12  % 毫秒时间戳
                try, t_dt = datetime(num/1000,'ConvertFrom','posixtime','TimeZone','UTC'); return; catch, end
            end
            if m > 1e8 && m < 2e9  % 秒时间戳
                try, t_dt = datetime(num,'ConvertFrom','posixtime','TimeZone','UTC'); return; catch, end
            end
            if m > 7e5 && m < 8.5e5  % MATLAB datenum
                try, t_dt = datetime(num,'ConvertFrom','datenum','TimeZone','UTC'); return; catch, end
            end

            ok = false;
        end
        
        function dt = parseTimeString(~, timeStr, baseDate)
            % 解析时间字符串 (HH:MM:SS 或 HH:MM)
            dt = NaT;
            timeStr = strtrim(timeStr);
            
            try
                % 尝试 HH:MM:SS 格式
                parts = split(timeStr, ':');
                if length(parts) == 3
                    h = str2double(parts{1});
                    m = str2double(parts{2});
                    s = str2double(parts{3});
                    dt = baseDate + hours(h) + minutes(m) + seconds(s);
                elseif length(parts) == 2
                    h = str2double(parts{1});
                    m = str2double(parts{2});
                    dt = baseDate + hours(h) + minutes(m);
                end
            catch
                dt = NaT;
            end
        end
    end

    methods (Access = public)
        function app = SatelliteSignalLabeler
            buildUI(app);
        end
    end
end