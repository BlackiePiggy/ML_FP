classdef CN0MultiPlotter < handle
    % CN0MultiPlotter: C/N0 时序多信号叠加绘图 GUI（纯代码）
    % 需求支持：
    % 1) 选择一个CSV文件
    % 2) 读取文件内容，形成卫星号列表（按从小到大排序）
    % 3) 选择其中一颗卫星，单独绘制当天0-24点的散点时序
    % 4) 图题包含：信号类型-卫星号 + "站点 日期"（如：S2W-G06 | cusv2070 2024-07-25）
    % 5) X轴覆盖当天 00:00:00 ~ 24:00:00，30 s 采样点
    % 6) 每次读取一个文件的一颗卫星一个信号，可加入缓存
    % 7) 从缓存多选信号，叠加绘制到同一张图（自动不同颜色）
    % 8) 从缓存任选 A、B 两条序列，绘制差分时序 A − B（共同时间点对齐）
    % 9) 支持从缓存移除所选条目
    %
    % 使用：app = CN0MultiPlotter;

    properties
        % UI
        UIFig
        Grid
        BtnPickFile
        FileEdit
        SatList
        SigList
        BtnAddCache
        CacheList
        BtnPlotOne
        BtnPlotMulti
        Ax

        % 差分与删除
        DropA
        DropB
        BtnDiffPlot
        BtnRemoveCache

        % 数据缓存
        T table               % 当前文件表
        CurrentFile string    % 当前文件路径
        Day0 datetime         % 当天起点（dateshift）
        Station string        % 站点（如 cusv2070）
        Cache struct          % 缓存数组
        Colors double         % 颜色表
        ColorIdx double       % 当前颜色索引
    end

    methods
        function app = CN0MultiPlotter()
            app.buildUI();
            app.Cache = struct('label',{},'time',{},'cn0',{},'sig',{},'sat',{},'station',{},'day0',{},'color',{});
            app.Colors = lines(20);
            app.ColorIdx = 1;
            try, app.UIFig.WindowState = 'maximized'; end %#ok<TRYNC>
        end

        function buildUI(app)
            app.UIFig = uifigure('Name','CN0 多信号叠加绘图','Position',[100 100 1280 720]);

            % 主网格：1行3列（左侧控制 / 中间绘图 / 右侧缓存）
            app.Grid = uigridlayout(app.UIFig,[1 3]);
            app.Grid.RowHeight   = {'1x'};
            app.Grid.ColumnWidth = {340, '1x', 380};
            app.Grid.Padding = [8 8 8 8];
            app.Grid.ColumnSpacing = 8;
            app.Grid.RowSpacing = 8;

            %% 左侧：控制区（可滚动）
            leftPanel = uipanel(app.Grid,'Title','数据与选择','Scrollable','on');
            leftPanel.Layout.Row = 1; leftPanel.Layout.Column = 1;

            gl = uigridlayout(leftPanel,[10 1]);
            gl.RowHeight   = {34,34, 22, 160, 22, 140, 40, 40, 40, '1x'};
            gl.ColumnWidth = {'1x'};
            gl.Padding = [8 8 8 8];
            gl.RowSpacing = 6;

            % 选择文件
            app.BtnPickFile = uibutton(gl,'Text','① 选择CSV文件','ButtonPushedFcn',@(~,~)app.onPickFile());
            app.BtnPickFile.Layout.Row = 1;

            app.FileEdit = uieditfield(gl,'text','Editable','off');
            app.FileEdit.Layout.Row = 2;
            app.FileEdit.Tooltip = "所选文件完整路径";

            % 卫星列表
            lblSat = uilabel(gl,'Text','卫星号（按编号升序）');
            lblSat.Layout.Row = 3;
            app.SatList = uilistbox(gl,'Multiselect','off');
            app.SatList.Layout.Row = 4;

            % 信号类型
            lblSig = uilabel(gl,'Text','信号类型（signal_code）');
            lblSig.Layout.Row = 5;
            app.SigList = uilistbox(gl,'Multiselect','off');
            app.SigList.Layout.Row = 6;

            % 操作按钮
            app.BtnAddCache  = uibutton(gl,'Text','② 加入缓存（当前文件+卫星+信号）','ButtonPushedFcn',@(~,~)app.onAddCache());
            app.BtnAddCache.Layout.Row = 7;

            app.BtnPlotOne   = uibutton(gl,'Text','③ 单星绘图（仅当前选择）','ButtonPushedFcn',@(~,~)app.onPlotOne());
            app.BtnPlotOne.Layout.Row = 8;

            spacer = uilabel(gl,'Text',''); %#ok<NASGU>
            spacer.Layout.Row = 9;

            %% 中间：大号绘图区（自适应）
            app.Ax = uiaxes(app.Grid);
            app.Ax.Layout.Row=1;
            app.Ax.Layout.Column=2;
            grid(app.Ax,'on'); box(app.Ax,'on');
            xlabel(app.Ax,'UTC 时间');
            ylabel(app.Ax,'C/N_0 (dB-Hz)');
            title(app.Ax,'CN0 时序图（0–24 h，30s采样）');
            % 注意：不要在这里设 xtickformat，等绘制后再设

            %% 右侧：缓存区（可滚动）
            rightPanel = uipanel(app.Grid,'Title','缓存区（叠加/差分/移除）','Scrollable','on');
            rightPanel.Layout.Row = 1; rightPanel.Layout.Column = 3;

            gr = uigridlayout(rightPanel,[9 1]);
            gr.RowHeight   = {22, 44, 22, '1x', 22, 28, 28, 40, 40};
            gr.ColumnWidth = {'1x'};
            gr.Padding = [8 8 8 8];
            gr.RowSpacing = 6;

            tip = uilabel(gr,'Text','条目：signal-sat | station date','HorizontalAlignment','left');
            tip.Layout.Row = 1;

            app.BtnPlotMulti = uibutton(gr,'Text','④ 叠加绘制（从下表多选）','ButtonPushedFcn',@(~,~)app.onPlotMulti());
            app.BtnPlotMulti.Layout.Row = 2;

            lblList = uilabel(gr,'Text','缓存列表（Ctrl/Shift 多选）：');
            lblList.Layout.Row = 3;

            app.CacheList = uilistbox(gr,'Multiselect','on');
            app.CacheList.Layout.Row = 4;
            app.CacheList.Items = {};
            app.CacheList.Value = {};

            lblDiff = uilabel(gr,'Text','差分设置：被减项 A  −  减项 B');
            lblDiff.Layout.Row = 5;

            app.DropA = uidropdown(gr,'Items',{'(空)'});
            app.DropA.Layout.Row = 6;

            app.DropB = uidropdown(gr,'Items',{'(空)'});
            app.DropB.Layout.Row = 7;

            app.BtnDiffPlot = uibutton(gr,'Text','⑤ 绘制差分 A − B','ButtonPushedFcn',@(~,~)app.onDiffPlot());
            app.BtnDiffPlot.Layout.Row = 8;

            app.BtnRemoveCache = uibutton(gr,'Text','⑥ 移除所选缓存项','ButtonPushedFcn',@(~,~)app.onRemoveCache());
            app.BtnRemoveCache.Layout.Row = 9;
        end

        %% 选择文件
        function onPickFile(app)
            [f,p] = uigetfile({'*.csv;*.txt','CSV/TXT 文件'},'选择CSV文件');
            if isequal(f,0), return; end
            app.CurrentFile = string(fullfile(p,f));
            app.FileEdit.Value = app.CurrentFile;

            % 读表（兼容逗号/空格/制表符）
            T = app.readFlexibleTable(app.CurrentFile);

            % 必要列名
            need = ["time_utc","sat","signal_code","station","CN0_dBHz"];
            for c = need
                if ~any(strcmpi(T.Properties.VariableNames, c))
                    uialert(app.UIFig,"缺少必要列："+c,'列缺失');
                    return;
                end
            end
            % 统一列名小写
            T.Properties.VariableNames = lower(T.Properties.VariableNames);

            % 解析时间
            T.time_utc = app.parseTimeCol(T.time_utc);
            if any(isnat(T.time_utc))
                uialert(app.UIFig,'时间列解析失败（存在 NaT），请检查格式','时间解析错误');
                return;
            end

            % 锁定当天 0–24h（以首条日期为准）
            d0 = dateshift(T.time_utc(1),'start','day');
            d1 = d0 + days(1);
            T = T(T.time_utc>=d0 & T.time_utc<=d1,:);

            % 站点（众数/首个）
            if iscellstr(T.station) || isstring(T.station)
                st = string(T.station);
                app.Station = app.modeString(st);
            else
                app.Station = "unknown";
            end

            % 卫星排序（按 PRN 数字）
            sats = unique(string(T.sat));
            [~,ord] = sort(app.prnNumber(sats));
            sats = sats(ord);
            app.SatList.Items = sats;
            app.SatList.Value = sats(1);

            % 信号类型
            sigs = unique(string(T.signal_code));
            app.SigList.Items = sigs;
            app.SigList.Value = sigs(1);

            % 保存
            app.T = T;
            app.Day0 = d0;

            % 清图但不设 XLim（避免数值轴与 datetime 冲突）
            cla(app.Ax);
            grid(app.Ax,'on'); box(app.Ax,'on');
            title(app.Ax,'文件已载入，请选择卫星与信号');
        end

        %% 加入缓存（当前文件 + 选中卫星 + 选中信号）
        function onAddCache(app)
            if isempty(app.T)
                uialert(app.UIFig,'请先选择并载入CSV文件','未载入文件'); return;
            end
            sat = string(app.SatList.Value);
            sig = string(app.SigList.Value);
            if strlength(sat)==0 || strlength(sig)==0
                uialert(app.UIFig,'请先在列表中选择卫星与信号','未选择'); return;
            end

            % 过滤出该卫星+信号的数据
            mask = string(app.T.sat)==sat & string(app.T.signal_code)==sig;
            if ~any(mask)
                uialert(app.UIFig,'当前文件中未找到该卫星与信号的数据','无数据'); return;
            end
            tt = app.T(mask,:);

            % 生成颜色
            c = app.Colors(app.ColorIdx,:);
            app.ColorIdx = app.ColorIdx + 1; if app.ColorIdx>size(app.Colors,1), app.ColorIdx=1; end

            % 生成标签
            label = sprintf('%s-%s | %s %s', sig, sat, app.Station, datestr(app.Day0,'yyyy-mm-dd'));

            % 存入缓存
            entry.label   = label;
            entry.time    = tt.time_utc;
            entry.cn0     = double(tt.cn0_dbhz);
            entry.sig     = sig;
            entry.sat     = sat;
            entry.station = app.Station;
            entry.day0    = app.Day0;
            entry.color   = c;

            app.Cache(end+1) = entry; %#ok<AGROW>
            app.refreshCacheList();

            uialert(app.UIFig,'已加入缓存，可在右侧列表多选叠加绘图','已加入缓存','Icon','success');
        end

        %% 单星绘图（仅当前选择，不必加入缓存）
        function onPlotOne(app)
            if isempty(app.T)
                uialert(app.UIFig,'请先选择并载入CSV文件','未载入文件');
                return;
            end
            sat = string(app.SatList.Value);
            sig = string(app.SigList.Value);
            if strlength(sat)==0 || strlength(sig)==0
                uialert(app.UIFig,'请先在列表中选择卫星与信号','未选择');
                return;
            end

            mask = string(app.T.sat)==sat & string(app.T.signal_code)==sig;
            if ~any(mask)
                uialert(app.UIFig,'当前文件中未找到该卫星与信号的数据','无数据');
                return;
            end
            tt = app.T(mask,:);

            cla(app.Ax); hold(app.Ax,'on');
            scatter(app.Ax, tt.time_utc, double(tt.cn0_dbhz), 20, 'filled');

            % 轴类型已是 datetime，再设置范围与刻度
            try
                xlim(app.Ax, [app.Day0, app.Day0+days(1)]);
                xtickformat(app.Ax,'HH:mm');
            catch
                xlim(app.Ax, [datenum(app.Day0), datenum(app.Day0+days(1))]);
                datetick(app.Ax,'x','HH:MM','keeplimits','keepticks');
            end
            grid(app.Ax,'on'); box(app.Ax,'on');
            xlabel(app.Ax,'UTC 时间'); ylabel(app.Ax,'C/N_0 (dB-Hz)');

            ttl = sprintf('%s-%s | %s %s', sig, sat, app.Station, datestr(app.Day0,'yyyy-mm-dd'));
            title(app.Ax, ttl);
            hold(app.Ax,'off');
        end

        %% 叠加绘图（从缓存多选）
        function onPlotMulti(app)
            sel = string(app.CacheList.Value);
            if isempty(sel)
                uialert(app.UIFig,'请在缓存区多选至少一个条目再绘图','未选择');
                return;
            end

            cla(app.Ax); hold(app.Ax,'on');
            legends = strings(0);
            drew = false;

            for i = 1:numel(app.Cache)
                if any(sel == string(app.Cache(i).label))
                    scatter(app.Ax, app.Cache(i).time, app.Cache(i).cn0, 20, app.Cache(i).color, 'filled');
                    legends(end+1) = string(app.Cache(i).label); %#ok<AGROW>
                    drew = true;
                end
            end

            if drew
                try
                    xlim(app.Ax, [app.Day0, app.Day0+days(1)]);
                    xtickformat(app.Ax,'HH:mm');
                catch
                    xlim(app.Ax, [datenum(app.Day0), datenum(app.Day0+days(1))]);
                    datetick(app.Ax,'x','HH:MM','keeplimits','keepticks');
                end
                grid(app.Ax,'on'); box(app.Ax,'on');
                xlabel(app.Ax,'UTC 时间'); ylabel(app.Ax,'C/N_0 (dB-Hz)');
                legend(app.Ax, legends, 'Interpreter','none','Location','best');
                title(app.Ax,'多信号叠加时序图（0–24 h，30s）');
            else
                title(app.Ax,'（无可绘制的所选缓存项）');
            end
            hold(app.Ax,'off');
        end

        %% 差分绘图（A − B，按共同时间点对齐）
        function onDiffPlot(app)
            if isempty(app.Cache)
                uialert(app.UIFig,'缓存为空，请先加入缓存后再做差分','无缓存'); return;
            end
            la = string(app.DropA.Value);
            lb = string(app.DropB.Value);
            if la == "(空)" || lb == "(空)"
                uialert(app.UIFig,'请在下拉框中选择“被减项 A”和“减项 B”','未选择'); return;
            end
            if la == lb
                uialert(app.UIFig,'A 和 B 不能相同','选择冲突'); return;
            end

            idxA = find(arrayfun(@(e) string(e.label)==la, app.Cache), 1);
            idxB = find(arrayfun(@(e) string(e.label)==lb, app.Cache), 1);
            if isempty(idxA) || isempty(idxB)
                uialert(app.UIFig,'所选缓存项不存在（请刷新缓存下拉）','未找到'); return;
            end
            A = app.Cache(idxA);
            B = app.Cache(idxB);

            % 对齐共同时间点（严格 30s 网格，不插值）
            try
                TT_A = timetable(A.time, A.cn0, 'VariableNames', {'cn0'});
                TT_B = timetable(B.time, B.cn0, 'VariableNames', {'cn0'});
                TT = synchronize(TT_A, TT_B, 'intersection');
                if isempty(TT)
                    uialert(app.UIFig,'两条时序没有共同时间点，无法做差分','无交集'); return;
                end
                diffy = TT.cn0_TT_A - TT.cn0_TT_B; % A − B
                t = TT.Time;
            catch
                [t, ia, ib] = intersect(A.time, B.time);
                if isempty(t)
                    uialert(app.UIFig,'两条时序没有共同时间点，无法做差分','无交集'); return;
                end
                diffy = A.cn0(ia) - B.cn0(ib);
            end

            cla(app.Ax); hold(app.Ax,'on');
            scatter(app.Ax, t, diffy, 20, 'filled');

            try
                xlim(app.Ax, [app.Day0, app.Day0+days(1)]);
                xtickformat(app.Ax,'HH:mm');
            catch
                xlim(app.Ax, [datenum(app.Day0), datenum(app.Day0+days(1))]);
                datetick(app.Ax,'x','HH:MM','keeplimits','keepticks');
            end
            grid(app.Ax,'on'); box(app.Ax,'on');
            xlabel(app.Ax,'UTC 时间');
            ylabel(app.Ax,'ΔC/N_0 (dB-Hz)');

            ttl = sprintf('差分：(%s) − (%s)', A.label, B.label);
            title(app.Ax, ttl, 'Interpreter','none');
            yline(app.Ax, 0, '--');
            hold(app.Ax,'off');
        end

        %% 移除缓存
        function onRemoveCache(app)
            sel = string(app.CacheList.Value);
            if isempty(sel)
                uialert(app.UIFig,'请先在“缓存列表”中选择要移除的条目','未选择'); return;
            end

            keep = true(1, numel(app.Cache));
            for i = 1:numel(app.Cache)
                if any(sel == string(app.Cache(i).label))
                    keep(i) = false;
                end
            end

            if all(keep)
                uialert(app.UIFig,'没有找到可移除的条目（可能列表已刷新）','未找到'); return;
            end

            app.Cache = app.Cache(keep);
            app.refreshCacheList();
        end

        %% —— 工具函数 —— %
        function refreshCacheList(app)
            labels = arrayfun(@(e)e.label, app.Cache, 'UniformOutput', false);
            app.CacheList.Items = labels;
            app.CacheList.Value = {}; % 默认不选

            % 同步差分下拉（A、B）
            if isempty(labels)
                app.DropA.Items = {'(空)'}; app.DropA.Value = '(空)';
                app.DropB.Items = {'(空)'}; app.DropB.Value = '(空)';
            else
                app.DropA.Items = labels;
                if ~ismember(app.DropA.Value, labels), app.DropA.Value = labels{1}; end
                app.DropB.Items = labels;
                if ~ismember(app.DropB.Value, labels)
                    app.DropB.Value = labels{min(2, numel(labels))};
                end
            end
        end

        function T = readFlexibleTable(~, filepath)
            % 尝试多种分隔方式：逗号/空格/制表符；文本统一为 string
            try
                opts = detectImportOptions(filepath,'NumHeaderLines',0,'TextType','string');
                Ttmp = readtable(filepath, opts);
                if width(Ttmp)==1
                    T = readtable(filepath, 'Delimiter',' \t', 'MultipleDelimsAsOne',true, 'TextType','string');
                else
                    T = Ttmp;
                end
                return;
            catch
            end
            % 退化方案1：逗号分隔
            try
                T = readtable(filepath,'Delimiter',',','TextType','string');
                if width(T)==1, error('only one col'); end
                return;
            catch
            end
            % 退化方案2：空格/制表符
            T = readtable(filepath,'Delimiter',' \t','MultipleDelimsAsOne',true,'TextType','string');
        end

        function dt = parseTimeCol(~, col)
            % 将 time_utc 列（string/cellstr/datetime）统一为 datetime（无时区）
            if isdatetime(col)
                dt = col; dt.TimeZone = ''; return;
            end
            s = string(col);

            % 修正形如 “00:00.0” -> “00:00:00.0”
            s = regexprep(s, '(\d{1,2}:\d{2})\.(\d+)$', '$1:00.$2');

            % 尝试若干格式
            fmts = [ ...
                "yyyy/M/d HH:mm:ss.S", ...
                "yyyy/M/d HH:mm:ss",   ...
                "yyyy/M/d HH:mm",      ...
                "yyyy/MM/dd HH:mm:ss.S", ...
                "yyyy/MM/dd HH:mm:ss", ...
                "yyyy/MM/dd HH:mm" ...
                ];

            dt = NaT(size(s));
            for k = 1:numel(fmts)
                mask = ismissing(dt);
                try
                    tmp = datetime(s(mask), 'InputFormat', fmts(k), 'TimeZone','');
                    dt(mask) = tmp;
                catch
                end
            end
            % 再兜底：自动解析
            bad = ismissing(dt);
            if any(bad)
                try
                    tmp = datetime(s(bad),'TimeZone','');
                    dt(bad) = tmp;
                catch
                end
            end

            % 将时间对齐到 30s 整点（向上对齐）
            sec = seconds(dt - dateshift(dt,'start','day'));
            r = mod(sec,30);
            fixmask = ~isnat(dt) & r~=0;
            dt(fixmask) = dt(fixmask) + seconds(30 - r(fixmask));
        end

        function m = modeString(~, x)
            u = unique(x);
            if numel(u)==1, m = u; return; end
            counts = zeros(size(u));
            for i=1:numel(u)
                counts(i) = sum(x==u(i));
            end
            [~,idx] = max(counts);
            m = u(idx);
        end

        function n = prnNumber(~, sats)
            % 从如 "G06" "G3" 中提取数字做排序键
            prn = regexp(sats,'\d+','match','once');
            n = zeros(size(sats));
            for i=1:numel(sats)
                v = str2double(prn{i});
                if isnan(v), v = inf; end
                n(i) = v;
            end
        end
    end
end
