classdef CN0LabelerApp < matlab.apps.AppBase
    % CN0LabelerApp: C/N0 时序标注工具（纯代码 GUI）
    % - 加载CSV（第1列时间，第2列C/N0），自动识别起止时间
    % - 右侧散点图：label=0 蓝色；label=1 红色
    % - 使用数据提示(DataTip)选点；拖动DataTip时实时更新选中点
    % - 设置“起始/结束时间”，对该单个区间一键标注为1或0
    %
    % 运行：app = CN0LabelerApp;

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

        % 右侧绘图
        RightPanel         matlab.ui.container.Panel
        Ax                 matlab.ui.control.UIAxes
    end

    properties (Access = private)
        % 数据
        T              datetime = datetime.empty
        Tnum           double   = []
        CN0            double   = []
        Label          double   = []   % 0/1

        % 绘图
        Sc             matlab.graphics.chart.primitive.Scatter = matlab.graphics.chart.primitive.Scatter.empty

        % 当前文件
        CurrentFile    string = ""

        % 选择点与候选区间
        SelectedIdx    double = NaN
        StartCand      datetime = NaT
        EndCand        datetime = NaT

        % DataTip 监听器（新版本支持）
        DTListener     event.listener = event.listener.empty

        % 拖动实时联动
        LastDTIndex    double = NaN
    end

    methods (Access = private)
        %% =============== UI 构建 ===============
        function buildUI(app)
            app.UIFigure = uifigure('Name','C/N0 Labeler','Position',[100 100 1200 700]);
            app.UIFigure.Color = [0.98 0.98 0.99];

            % 鼠标移动时，若存在 DataTip，则实时同步选中点
            app.UIFigure.WindowButtonMotionFcn = @(~,~)app.onFigureMotion();

            app.Grid = uigridlayout(app.UIFigure,[1,2]);
            app.Grid.ColumnWidth = {380,'1x'};
            app.Grid.RowHeight = {'1x'};

            %% 左侧
            app.LeftPanel = uipanel(app.Grid,'Title','控制面板');
            app.LeftPanel.FontWeight = 'bold';
            gl = uigridlayout(app.LeftPanel,[10,3]);
            gl.RowHeight = repmat({28},1,10);
            gl.RowSpacing = 6;
            gl.ColumnWidth = {90,'1x',70};

            % 输入文件夹
            app.InputFolderLabel = uilabel(gl,'Text','输入文件夹:','FontWeight','bold');
            app.InputFolderLabel.Layout.Row = 1; app.InputFolderLabel.Layout.Column = 1;
            app.InputFolderEdit = uieditfield(gl,'text','Value',pwd);
            app.InputFolderEdit.Layout.Row = 1; app.InputFolderEdit.Layout.Column = 2;
            app.PickInputBtn = uibutton(gl,'Text','选择','ButtonPushedFcn',@(~,~)app.pickInputFolder());
            app.PickInputBtn.Layout.Row = 1; app.PickInputBtn.Layout.Column = 3;

            % 输出文件夹
            app.OutputFolderLabel = uilabel(gl,'Text','输出文件夹:','FontWeight','bold');
            app.OutputFolderLabel.Layout.Row = 2; app.OutputFolderLabel.Layout.Column = 1;
            app.OutputFolderEdit = uieditfield(gl,'text','Value',pwd);
            app.OutputFolderEdit.Layout.Row = 2; app.OutputFolderEdit.Layout.Column = 2;
            app.PickOutputBtn = uibutton(gl,'Text','选择','ButtonPushedFcn',@(~,~)app.pickOutputFolder());
            app.PickOutputBtn.Layout.Row = 2; app.PickOutputBtn.Layout.Column = 3;

            % 载入/保存
            app.LoadBtn = uibutton(gl,'Text','载入CSV','ButtonPushedFcn',@(~,~)app.onLoadFile());
            app.LoadBtn.Layout.Row = 3; app.LoadBtn.Layout.Column = [1 2];
            app.SaveBtn = uibutton(gl,'Text','保存标注','ButtonPushedFcn',@(~,~)app.onSave());
            app.SaveBtn.Layout.Row = 3; app.SaveBtn.Layout.Column = 3;

            % 文件&起止时间
            app.FileLabel = uilabel(gl,'Text','当前文件: -','Tooltip','显示已载入文件名');
            app.FileLabel.Layout.Row = 4; app.FileLabel.Layout.Column = [1 3];

            app.StartTimeLabel = uilabel(gl,'Text','数据起始: -');
            app.StartTimeLabel.Layout.Row = 5; app.StartTimeLabel.Layout.Column = [1 3];
            app.EndTimeLabel = uilabel(gl,'Text','数据结束: -');
            app.EndTimeLabel.Layout.Row = 6; app.EndTimeLabel.Layout.Column = [1 3];

            % 选点信息 & 起止设置
            app.SelectedInfoLabel = uilabel(gl,'Text','选中点: -','FontWeight','bold');
            app.SelectedInfoLabel.Layout.Row = 7; app.SelectedInfoLabel.Layout.Column = [1 3];

            % ——把第 8 行行高适当增大——
            gl.RowHeight{8} = 30;

            % ——第 8 行：两个按钮平分——
            subgrid = uigridlayout(gl,[1,2]);
            subgrid.Layout.Row = 8;
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

            app.Mark1Btn = uibutton(gl,'Text','标注为 1（启用）','ButtonPushedFcn',@(~,~)app.onMarkCurrent(1));
            app.Mark1Btn.Layout.Row = 9; app.Mark1Btn.Layout.Column = [1 3];

            app.Mark0Btn = uibutton(gl,'Text','标注为 0（关闭）','ButtonPushedFcn',@(~,~)app.onMarkCurrent(0));
            app.Mark0Btn.Layout.Row = 10; app.Mark0Btn.Layout.Column = [1 3];

            %% 右侧绘图
            app.RightPanel = uipanel(app.Grid,'Title','时序可视化');
            app.RightPanel.FontWeight = 'bold';
            app.Ax = uiaxes(app.RightPanel,'Position',[30 30 740 620]);

            try enableDefaultInteractivity(app.Ax); catch, end
            try app.Ax.Toolbar.Visible = 'on'; catch, end

            grid(app.Ax,'on');
            xlabel(app.Ax,'Time');
            ylabel(app.Ax,'C/N0');
            title(app.Ax,'C/N0 Timeseries (蓝:0, 红:1)');
        end

        %% =============== 事件回调 ===============
        function pickInputFolder(app)
            p = uigetdir(app.InputFolderEdit.Value,'选择默认输入文件夹');
            if ischar(p) || (isstring(p) && strlength(p)>0)
                app.InputFolderEdit.Value = string(p);
            end
        end

        function pickOutputFolder(app)
            p = uigetdir(app.OutputFolderEdit.Value,'选择默认输出文件夹');
            if ischar(p) || (isstring(p) && strlength(p)>0)
                app.OutputFolderEdit.Value = string(p);
            end
        end

        function onLoadFile(app)
            startDir = app.InputFolderEdit.Value;
            if ~(isstring(startDir) || ischar(startDir)) || ~isfolder(startDir)
                startDir = pwd;
            end
            [f,p] = uigetfile({'*.csv','CSV Files (*.csv)'},'选择C/N0 CSV文件',startDir);
            if isequal(f,0); return; end
            app.CurrentFile = string(fullfile(p,f));
            app.FileLabel.Text = "当前文件: " + f;

            % 读表 & 时间解析
            try
                Ttbl = readtable(app.CurrentFile);
                if size(Ttbl,2) < 2
                    uialert(app.UIFigure,'CSV 至少需要两列：时间、C/N0','列数不足'); return;
                end
                tcol = Ttbl{:,1};
                ccol = Ttbl{:,2};
                [t_dt, ok] = app.parseTimeColumn(tcol);
                if ~ok
                    uialert(app.UIFigure,'无法识别时间列格式（支持 datetime/epoch秒/epoch毫秒/datenum）','时间解析失败'); return;
                end
                app.T = t_dt(:);
                app.Tnum = datenum(app.T);   % 使用 numeric 画散点（避免类型不匹配）
                app.CN0 = double(ccol(:));
                n = numel(app.CN0);
                app.Label = zeros(n,1);

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
            [inPath, inName, ~] = fileparts(app.CurrentFile);
            outFolder = string(app.OutputFolderEdit.Value);
            if ~isfolder(outFolder); outFolder = inPath; end
            outFile = fullfile(outFolder, inName + "_labeled.csv");

            Tout = table(app.T, app.CN0, app.Label, 'VariableNames', {'Time','CN0','Label'});
            try
                writetable(Tout, outFile);
                uialert(app.UIFigure, "已保存: " + outFile, '保存成功');
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
            app.Label(mask) = val;
            app.refreshColors();
        end

        %% =============== 绘图/颜色/数据提示 ===============
        function plotScatter(app)
            cla(app.Ax);
            if isempty(app.Tnum); return; end

            % 初始全蓝
            c = repmat([0 0.447 0.741], numel(app.Tnum),1); % MATLAB 默认蓝
            app.Sc = scatter(app.Ax, app.Tnum, app.CN0, 12, c, 'filled');

            % X 轴显示为时间
            datetick(app.Ax,'x','keeplimits');
            title(app.Ax,'C/N0 Timeseries (蓝:0, 红:1)');
            xlabel(app.Ax,'Time');
            ylabel(app.Ax,'C/N0');
            grid(app.Ax,'on');

            % 配置 DataTip 显示内容（Time / CN0 / Label）
            try
                row1 = dataTipTextRow('Time', string(app.T));
                row2 = dataTipTextRow('C/N0', app.CN0);
                row3 = dataTipTextRow('Label', app.Label);
                app.Sc.DataTipTemplate.DataTipRows = [row1 row2 row3];
            catch
            end

            % 允许点击散点来创建/移动 DataTip（兼容老版本）
            app.Sc.PickableParts = 'all';
            app.Sc.HitTest = 'on';
            app.Sc.ButtonDownFcn = @(src,event)app.onScatterClick(event);

            % 新版：监听 DataTipCreated（拖动/创建都会触发一次）
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
                    row2 = dataTipTextRow('C/N0', app.CN0);
                    row3 = dataTipTextRow('Label', app.Label);
                    app.Sc.DataTipTemplate.DataTipRows = [row1 row2 row3];
                end
            catch
            end

            drawnow limitrate;
        end

        %% =============== DataTip 交互 ===============
        function onDataTipCreated(app, evt)
            % 新版事件：创建/拖动结束时触发一次
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
            % 鼠标移动时：若有 DataTip，则实时同步选中点（拖动时会不断变化）
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
            % 兜底：点击散点创建/移动 DataTip，并立即同步
            if isempty(app.Tnum); return; end

            % 点击位置对应的轴坐标
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
                        dts(1).Target = app.Sc; % 确保目标对
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
            % 优先读 DataIndex；若无则用 Position 反算
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
                pos = dt.Position; % [x y z]
                x = pos(1); y = pos(2);
                X = app.Sc.XData; Y = app.Sc.YData;
                [~, idx] = min(abs(X - x) + abs(Y - y));
            catch
                idx = NaN;
            end
        end

        function dt = getSingleDataTip(app)
            % 取当前轴中的单个 datatip（若多个只留第一个）
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
            if isempty(app.CN0) || idx<1 || idx>numel(app.CN0)
                return;
            end
            app.SelectedIdx = idx;
            app.SelectedInfoLabel.Text = sprintf('选中点: %s | C/N0=%.3f | Label=%d', ...
                string(app.T(idx)), app.CN0(idx), app.Label(idx));
            app.flashPoint(idx);
        end

        function flashPoint(app, idx)
            % 简单闪烁效果提示所选点（含越界保护）
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

        %% =============== 时间解析 ===============
        function [t_dt, ok] = parseTimeColumn(~, tcol)
            ok = true; t_dt = datetime.empty;

            if isdatetime(tcol)
                t_dt = tcol; return;
            end

            if iscellstr(tcol) || isstring(tcol) || ischar(tcol)
                try
                    t_dt = datetime(tcol, 'InputFormat','', 'TimeZone','local'); return;
                catch
                end
            end

            try
                num = double(tcol);
            catch
                ok = false; return;
            end

            m = median(num,'omitnan');

            if m > 1e11 && m < 2e12
                try, t_dt = datetime(num/1000,'ConvertFrom','posixtime','TimeZone','local'); return; catch, end
            end
            if m > 1e8 && m < 2e9
                try, t_dt = datetime(num,'ConvertFrom','posixtime','TimeZone','local'); return; catch, end
            end
            if m > 7e5 && m < 8.5e5
                try, t_dt = datetime(num,'ConvertFrom','datenum','TimeZone','local'); return; catch, end
            end

            try
                t_dt = datetime(string(tcol),'InputFormat','', 'TimeZone','local'); return;
            catch
                ok = false;
            end
        end
    end

    methods (Access = public)
        function app = CN0LabelerApp
            buildUI(app);
        end
    end
end
