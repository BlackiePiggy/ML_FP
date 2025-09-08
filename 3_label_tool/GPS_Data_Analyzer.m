function GPS_Data_Analyzer
    % GPS数据时序分析GUI程序
    % 支持多测站、多卫星、多DOY数据的加载和可视化
    % 新增数据标注功能
    % 优化：只加载用户选中的文件
    
    % 全局变量存储数据
    global data_storage;
    data_storage = struct();
    data_storage.files = {};
    data_storage.data = {};
    data_storage.stations = {};
    data_storage.satellites = {};
    data_storage.doys = {};
    data_storage.annotations = {}; % 存储标注信息
    data_storage.current_start_point = [];
    data_storage.current_end_point = [];
    data_storage.annotation_mode = false;
    
    % 创建主窗口
    fig = figure('Name', 'GPS数据时序分析器', 'Position', [0, 0, 1600, 900], ...
                 'MenuBar', 'none', 'ToolBar', 'figure', 'Resize', 'on');
    
    % 创建UI控件
    createUI(fig);
end

function createUI(fig)
    global data_storage;

    % ====== 主容器 ======
    main_panel = uipanel('Parent', fig, 'Position', [0, 0, 1, 1], 'BorderType', 'none');

    % 左右两列：左侧控制区 0.34，右侧绘图区 0.65，中间预留 0.01 缝隙
    control_panel = uipanel('Parent', main_panel, ...
        'Units','normalized', 'Position', [0.01, 0.02, 0.32, 0.96], ...
        'Title', '数据控制面板', 'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.95, 0.95, 0.95]);

    plot_panel = uipanel('Parent', main_panel, ...
        'Units','normalized', 'Position', [0.34, 0.02, 0.65, 0.96], ...
        'Title', '数据可视化', 'FontSize', 12, 'FontWeight', 'bold');

    % ====== 左侧四个分区：用"自顶向下堆叠"的参数化布局（避免重叠） ======
    left = 0.03;          % 子面板左边距
    width = 0.94;         % 子面板宽度
    top_margin = 0.02;    % 顶部留白
    bottom_margin = 0.02; % 底部留白
    gap = 0.015;          % 子面板之间的垂直间距

    % 调整高度分配：文件选择减少，数据选择增加
    h_file    = 0.08; % 文件选择（减少）
    h_select  = 0.46; % 数据选择（增加，包含加载按钮）
    h_annot   = 0.20; % 数据标注
    h_plotctl = 0.22; % 绘图控制

    total = h_file + h_select + h_annot + h_plotctl + 3*gap + top_margin + bottom_margin;
    if total > 1
        % 如果调得太满，这里做一次自动压缩，防止重叠
        scale = (1 - top_margin - bottom_margin - 3*gap) / (h_file + h_select + h_annot + h_plotctl);
        h_file    = h_file    * scale;
        h_select  = h_select  * scale;
        h_annot   = h_annot   * scale;
        h_plotctl = h_plotctl * scale;
    end

    % 自顶向下计算每个子面板的位置
    y_top = 1 - top_margin;

    y_file    = y_top - h_file;
    y_select  = y_file - gap - h_select;
    y_annot   = y_select - gap - h_annot;
    y_plotctl = y_annot - gap - h_plotctl;

    % ====== 文件选择区（简化） ======
    file_panel = uipanel('Parent', control_panel, 'Units','normalized', ...
        'Position', [left, y_file, width, h_file], ...
        'Title', '文件扫描', 'FontWeight', 'bold', 'BackgroundColor', [0.98, 0.98, 1]);

    uicontrol('Parent', file_panel, 'Style', 'pushbutton', 'String', '选择数据文件夹', ...
        'Units','normalized','Position', [0.03, 0.45, 0.45, 0.45], 'FontSize', 10, ...
        'BackgroundColor', [0.3, 0.6, 0.9], 'ForegroundColor', 'white', ...
        'Callback', @selectDataFolder);

    data_storage.status_text = uicontrol('Parent', file_panel, 'Style', 'text', ...
        'String', '请选择包含GPS数据的文件夹', 'Units','normalized', ...
        'Position', [0.50, 0.45, 0.47, 0.45], 'FontSize', 8, ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98, 0.98, 1]);

    data_storage.scan_info = uicontrol('Parent', file_panel, 'Style', 'text', ...
        'String', '', 'Units','normalized', ...
        'Position', [0.03, 0.05, 0.94, 0.35], 'FontSize', 8, ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98, 0.98, 1]);

    % ====== 数据选择区（扩展，包含加载功能） ======
    select_panel = uipanel('Parent', control_panel, 'Units','normalized', ...
        'Position', [left, y_select, width, h_select], ...
        'Title', '数据选择与加载', 'FontWeight', 'bold', 'BackgroundColor', [0.98, 1, 0.98]);

    % 四个选择子区：三个选择框 + 一个参数控制 + 加载按钮区
    row_gap = 0.02;
    load_area_h = 0.12; % 加载按钮区域高度
    param_area_h = 0.08; % 参数选择区域高度
    select_area_h = (1 - load_area_h - param_area_h - 5*row_gap) / 3; % 三个选择框均分

    y_load = 1 - row_gap - load_area_h;
    y_station = y_load - row_gap - select_area_h;
    y_satellite = y_station - row_gap - select_area_h;
    y_doy = y_satellite - row_gap - select_area_h;
    y_param = y_doy - row_gap - param_area_h;

    % 加载控制区
    load_control_panel = uipanel('Parent', select_panel, 'Units','normalized', ...
        'Position', [0.02, y_load, 0.96, load_area_h], 'Title', '数据加载', ...
        'FontWeight', 'bold', 'FontSize', 9, 'BackgroundColor', [0.95, 1, 0.95]);

    uicontrol('Parent', load_control_panel, 'Style', 'pushbutton', 'String', '加载选中文件', ...
        'Units','normalized','Position', [0.03, 0.40, 0.30, 0.50], 'FontSize', 10, ...
        'FontWeight', 'bold', 'BackgroundColor', [0.2, 0.7, 0.3], 'ForegroundColor', 'white', ...
        'Callback', @loadSelectedFiles);

    uicontrol('Parent', load_control_panel, 'Style', 'pushbutton', 'String', '清空已加载数据', ...
        'Units','normalized','Position', [0.35, 0.40, 0.30, 0.50], 'FontSize', 9, ...
        'BackgroundColor', [0.8, 0.4, 0.2], 'ForegroundColor', 'white', ...
        'Callback', @clearLoadedData);

    data_storage.load_status_text = uicontrol('Parent', load_control_panel, 'Style', 'text', ...
        'String', '未加载数据', 'Units','normalized', ...
        'Position', [0.67, 0.40, 0.30, 0.50], 'FontSize', 8, ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.95, 1, 0.95]);

    % 测站选择
    station_panel = uipanel('Parent', select_panel, 'Units','normalized', ...
        'Position', [0.02, y_station, 0.96, select_area_h], 'Title', '测站选择', ...
        'FontWeight', 'bold', 'FontSize', 9, 'BackgroundColor', [1, 0.98, 0.98]);

    data_storage.station_listbox = uicontrol('Parent', station_panel, 'Style', 'listbox', ...
        'Units','normalized','Position', [0.03, 0.10, 0.82, 0.80], 'Max', 50, ...
        'String', {}, 'Value', [], 'FontSize', 8);

    uicontrol('Parent', station_panel, 'Style', 'pushbutton', 'String', '全选', ...
        'Units','normalized','Position', [0.87, 0.58, 0.10, 0.30], 'FontSize', 8, ...
        'Callback', @(~,~) selectAllItems(data_storage.station_listbox));
    uicontrol('Parent', station_panel, 'Style', 'pushbutton', 'String', '清空', ...
        'Units','normalized','Position', [0.87, 0.18, 0.10, 0.30], 'FontSize', 8, ...
        'Callback', @(~,~) set(data_storage.station_listbox, 'Value', []));

    % 卫星选择
    satellite_panel = uipanel('Parent', select_panel, 'Units','normalized', ...
        'Position', [0.02, y_satellite, 0.96, select_area_h], 'Title', '卫星选择', ...
        'FontWeight', 'bold', 'FontSize', 9, 'BackgroundColor', [0.98, 0.98, 1]);

    data_storage.satellite_listbox = uicontrol('Parent', satellite_panel, 'Style', 'listbox', ...
        'Units','normalized','Position', [0.03, 0.10, 0.82, 0.80], 'Max', 50, ...
        'String', {}, 'Value', [], 'FontSize', 8);

    uicontrol('Parent', satellite_panel, 'Style', 'pushbutton', 'String', '全选', ...
        'Units','normalized','Position', [0.87, 0.58, 0.10, 0.30], 'FontSize', 8, ...
        'Callback', @(~,~) selectAllItems(data_storage.satellite_listbox));
    uicontrol('Parent', satellite_panel, 'Style', 'pushbutton', 'String', '清空', ...
        'Units','normalized','Position', [0.87, 0.18, 0.10, 0.30], 'FontSize', 8, ...
        'Callback', @(~,~) set(data_storage.satellite_listbox, 'Value', []));

    % DOY选择
    doy_panel = uipanel('Parent', select_panel, 'Units','normalized', ...
        'Position', [0.02, y_doy, 0.96, select_area_h], 'Title', 'DOY选择 (年-天)', ...
        'FontWeight', 'bold', 'FontSize', 9, 'BackgroundColor', [0.98, 1, 0.98]);

    data_storage.doy_listbox = uicontrol('Parent', doy_panel, 'Style', 'listbox', ...
        'Units','normalized','Position', [0.03, 0.10, 0.82, 0.80], 'Max', 50, ...
        'String', {}, 'Value', [], 'FontSize', 8);

    uicontrol('Parent', doy_panel, 'Style', 'pushbutton', 'String', '全选', ...
        'Units','normalized','Position', [0.87, 0.58, 0.10, 0.30], 'FontSize', 8, ...
        'Callback', @(~,~) selectAllItems(data_storage.doy_listbox));
    uicontrol('Parent', doy_panel, 'Style', 'pushbutton', 'String', '清空', ...
        'Units','normalized','Position', [0.87, 0.18, 0.10, 0.30], 'FontSize', 8, ...
        'Callback', @(~,~) set(data_storage.doy_listbox, 'Value', []));

    % 参数选择
    param_panel = uipanel('Parent', select_panel, 'Units','normalized', ...
        'Position', [0.02, y_param, 0.96, param_area_h], 'BorderType', 'none', ...
        'BackgroundColor', [0.98, 1, 0.98]);
    uicontrol('Parent', param_panel, 'Style', 'text', 'String', '绘图参数:', ...
        'Units','normalized', 'Position', [0.02, 0.30, 0.20, 0.60], 'FontWeight', 'bold', ...
        'FontSize', 9, 'BackgroundColor', [0.98, 1, 0.98], 'HorizontalAlignment','left');

    data_storage.param_dropdown = uicontrol('Parent', param_panel, 'Style', 'popupmenu', ...
        'String', {'S1C', 'S2W', 'S2W_S1C_diff', 'elevation', 'azimuth', 'slant_range'}, ...
        'Units','normalized','Position', [0.22, 0.30, 0.35, 0.60], 'FontSize', 9);

    % ====== 数据标注区 ======
    annotation_panel = uipanel('Parent', control_panel, 'Units','normalized', ...
        'Position', [left, y_annot, width, h_annot], ...
        'Title', '数据标注', 'FontWeight', 'bold', 'BackgroundColor', [1, 0.95, 0.9]);

    uicontrol('Parent', annotation_panel, 'Style', 'pushbutton', 'String', '设为起始点', ...
        'Units','normalized','Position', [0.03, 0.8, 0.22, 0.2], 'FontSize', 9, ...
        'BackgroundColor', [0.2, 0.8, 0.2], 'ForegroundColor', 'white', ...
        'Callback', @setStartPoint);

    uicontrol('Parent', annotation_panel, 'Style', 'pushbutton', 'String', '设为结束点', ...
        'Units','normalized','Position', [0.27, 0.8, 0.22, 0.2], 'FontSize', 9, ...
        'BackgroundColor', [0.8, 0.2, 0.2], 'ForegroundColor', 'white', ...
        'Callback', @setEndPoint);

    uicontrol('Parent', annotation_panel, 'Style', 'pushbutton', 'String', '标注为 1', ...
        'Units','normalized','Position', [0.51, 0.8, 0.20, 0.2], 'FontSize', 9, ...
        'BackgroundColor', [0.1, 0.4, 0.8], 'ForegroundColor', 'white', ...
        'Callback', @(~,~) annotateRegion(1));

    uicontrol('Parent', annotation_panel, 'Style', 'pushbutton', 'String', '标注为 0', ...
        'Units','normalized','Position', [0.73, 0.8, 0.20, 0.2], 'FontSize', 9, ...
        'BackgroundColor', [0.6, 0.6, 0.6], 'ForegroundColor', 'white', ...
        'Callback', @(~,~) annotateRegion(0));

    data_storage.annotation_info = uicontrol('Parent', annotation_panel, 'Style', 'text', ...
        'String', '请先用数据游标选择点，然后设置起始点和结束点', ...
        'Units','normalized','Position', [0.03, 0.55, 0.94, 0.22], 'FontSize', 8, ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [1, 0.95, 0.9]);

    uicontrol('Parent', annotation_panel, 'Style', 'pushbutton', 'String', '选择输出文件夹', ...
        'Units','normalized','Position', [0.03, 0.1, 0.30, 0.2], 'FontSize', 9, ...
        'BackgroundColor', [0.8, 0.4, 0.8], 'ForegroundColor', 'white', ...
        'Callback', @selectOutputFolder);

    uicontrol('Parent', annotation_panel, 'Style', 'pushbutton', 'String', '导出标注数据', ...
        'Units','normalized','Position', [0.35, 0.1, 0.30, 0.2], 'FontSize', 9, ...
        'BackgroundColor', [0.8, 0.6, 0.2], 'ForegroundColor', 'white', ...
        'Callback', @exportAnnotatedData);

    uicontrol('Parent', annotation_panel, 'Style', 'pushbutton', 'String', '清除所有标注', ...
        'Units','normalized','Position', [0.67, 0.1, 0.30, 0.2], 'FontSize', 9, ...
        'BackgroundColor', [0.9, 0.3, 0.3], 'ForegroundColor', 'white', ...
        'Callback', @clearAllAnnotations);

    data_storage.output_path_text = uicontrol('Parent', annotation_panel, 'Style', 'text', ...
        'String', '输出路径: 未设置', 'Units','normalized', ...
        'Position', [0.03, 0.5, 0.94, 0.1], 'FontSize', 8, ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [1, 0.95, 0.9]);

    % ====== 绘图控制区 ======
    plot_control_panel = uipanel('Parent', control_panel, 'Units','normalized', ...
        'Position', [left, y_plotctl, width, h_plotctl], ...
        'Title', '绘图控制', 'FontWeight', 'bold', 'BackgroundColor', [1, 0.98, 0.95]);

    uicontrol('Parent', plot_control_panel, 'Style', 'pushbutton', 'String', '绘制时间序列', ...
        'Units','normalized','Position', [0.03, 0.78, 0.25, 0.20], 'FontSize', 11, ...
        'FontWeight', 'bold', 'BackgroundColor', [0.2, 0.7, 0.2], 'ForegroundColor', 'white', ...
        'Callback', @plotTimeSeries);

    uicontrol('Parent', plot_control_panel, 'Style', 'pushbutton', 'String', '清除图形', ...
        'Units','normalized','Position', [0.47, 0.78, 0.25, 0.20], 'FontSize', 10, ...
        'BackgroundColor', [0.8, 0.3, 0.3], 'ForegroundColor', 'white', ...
        'Callback', @clearPlots);

    options_panel = uipanel('Parent', plot_control_panel, 'Units','normalized', ...
        'Position', [0.03, 0.5, 0.94, 0.3], 'Title', '显示选项', 'FontSize', 10, ...
        'BackgroundColor', [1, 0.98, 0.95]);

    data_storage.legend_checkbox = uicontrol('Parent', options_panel, 'Style', 'checkbox', ...
        'String', '显示图例', 'Units','normalized','Position', [0.03, 0.20, 0.22, 0.60], ...
        'Value', 1, 'FontSize', 9, 'BackgroundColor', [1, 0.98, 0.95]);

    data_storage.grid_checkbox = uicontrol('Parent', options_panel, 'Style', 'checkbox', ...
        'String', '显示网格', 'Units','normalized','Position', [0.27, 0.20, 0.22, 0.60], ...
        'Value', 1, 'FontSize', 9, 'BackgroundColor', [1, 0.98, 0.95]);

    data_storage.cursor_checkbox = uicontrol('Parent', options_panel, 'Style', 'checkbox', ...
        'String', '启用数据游标', 'Units','normalized','Position', [0.51, 0.20, 0.28, 0.60], ...
        'Value', 1, 'FontSize', 9, 'BackgroundColor', [1, 0.98, 0.95], ...
        'Callback', @toggleDataCursor);

    data_storage.annotation_display_checkbox = uicontrol('Parent', options_panel, 'Style', 'checkbox', ...
        'String', '显示标注区域', 'Units','normalized','Position', [0.81, 0.20, 0.16, 0.60], ...
        'Value', 1, 'FontSize', 9, 'BackgroundColor', [1, 0.98, 0.95], ...
        'Callback', @toggleAnnotationDisplay);

    stats_panel = uipanel('Parent', plot_control_panel, 'Units','normalized', ...
        'Position', [0.03, 0.15, 0.94, 0.35], 'Title', '统计信息', 'FontSize', 10, ...
        'BackgroundColor', [1, 0.98, 0.95]);

    data_storage.stats_text = uicontrol('Parent', stats_panel, 'Style', 'text', ...
        'String', '统计信息将在此显示', 'Units','normalized','Position', [0.03, 0.10, 0.94, 0.80], ...
        'FontSize', 9, 'HorizontalAlignment', 'left', 'BackgroundColor', [1, 0.98, 0.95]);

    % ====== 右侧绘图区 ======
    data_storage.axes = axes('Parent', plot_panel, 'Units','normalized', ...
        'Position', [0.08, 0.08, 0.88, 0.84]);
    title(data_storage.axes, '请选择数据并点击绘制时间序列', 'FontSize', 14);
    xlabel(data_storage.axes, '时间 (UTC)', 'FontSize', 12);
    ylabel(data_storage.axes, '数值', 'FontSize', 12);
    grid(data_storage.axes, 'on');

    % 保存图窗句柄，供其它函数使用
    data_storage.fig = fig;
    
    % 数据游标：一定要把 figure 句柄传给 datacursormode
    data_storage.dcm = datacursormode(data_storage.fig);
    set(data_storage.dcm, 'UpdateFcn', @dataCursorUpdateFcn);
    datacursormode(data_storage.fig, 'on');

    % 其它初始化
    data_storage.output_folder = '';
    data_storage.last_cursor_point = [];
    data_storage.folder_path = '';  % 添加存储文件夹路径
end


function txt = dataCursorUpdateFcn(~, event_obj)
    % 从被拾取的散点图句柄回溯到原始表行，读取原始 time_utc（含年月日）与 t_abs
    global data_storage;
    
    h = get(event_obj, 'Target');   % 被点击的散点图句柄
    idx = [];                       % 该点在散点中的索引
    if isprop(event_obj, 'DataIndex')
        try
            idx = event_obj.DataIndex;
        catch
            idx = [];
        end
    end
    
    % 如果拿不到 DataIndex，退化为用最近点匹配
    if isempty(idx)
        try
            pos = get(event_obj, 'Position');     % [x,y]（x可能是datetime）
            xd  = get(h, 'XData');                % 绘图用的datetime/datenum
            if isdatetime(xd)
                xq = pos(1);
                [~, idx] = min(abs(xd - xq));
            else
                xq = pos(1);
                [~, idx] = min(abs(xd - xq));
            end
        catch
            idx = 1;
        end
    end
    
    % 取回绑定在该散点图上的来源信息
    file_idx = getappdata(h, 'file_idx');
    row_idx  = getappdata(h, 'row_idx');
    if isempty(file_idx) || isempty(row_idx) || idx < 1 || idx > numel(row_idx)
        % 兜底：无法映射时仍显示当前位置
        pos = get(event_obj, 'Position');
        if isdatetime(pos(1))
            time_dt = pos(1);
        else
            time_dt = datetime(pos(1), 'ConvertFrom','datenum', 'TimeZone','UTC');
        end
        time_abs = posixtime(time_dt);
        y_val = pos(2);
        param_idx = get(data_storage.param_dropdown, 'Value');
        param_names = get(data_storage.param_dropdown, 'String');
        param_name = param_names{param_idx};
        txt = {['时间: ', datestr(time_dt, 'yyyy-mm-dd HH:MM:SS')], ...
               [param_name, ': ', num2str(y_val, '%.3f')], ...
               ['绝对秒: ', num2str(time_abs, '%.3f')]};
        data_storage.last_cursor_point = struct('time_dt', time_dt, 'time_abs', time_abs, 'value', y_val, 'target', h);
        if isprop(h,'DisplayName') && ~isempty(get(h,'DisplayName'))
            txt{end+1} = ['数据源: ', get(h,'DisplayName')];
        end
        txt{end+1} = '点击起始点/结束点按钮进行标注';
        return;
    end
    
    % 映射到原表的真实行
    row = row_idx(idx);
    file_info  = data_storage.data{file_idx};
    T          = file_info.data;
    
    % 原始时间：严格取自原始 data 表
    time_dt = T.time_utc(row);
    if isempty(time_dt.TimeZone)
        time_dt.TimeZone = 'UTC';
    end
    if ismember('t_abs', T.Properties.VariableNames)
        time_abs = double(T.t_abs(row));
    else
        time_abs = posixtime(time_dt);
    end
    
    % 取当前参数的数值（与图上 value 一致）
    param_idx = get(data_storage.param_dropdown, 'Value');
    param_names = get(data_storage.param_dropdown, 'String');
    param_name = param_names{param_idx};
    y_val = NaN;
    if ismember(param_name, T.Properties.VariableNames)
        y_val = T.(param_name)(row);
    else
        % 兜底：用图上 y
        pos = get(event_obj, 'Position');
        y_val = pos(2);
    end
    
    % 更新 last_cursor_point（后续标注都用这里的 time_abs/time_dt）
    data_storage.last_cursor_point = struct( ...
        'time_dt', time_dt, ...
        'time_abs', time_abs, ...
        'value', y_val, ...
        'target', h ...
    );
    
    % 组装显示文本
    txt = {['时间: ', datestr(time_dt, 'yyyy-mm-dd HH:MM:SS')], ...
           [param_name, ': ', num2str(y_val, '%.3f')], ...
           ['绝对秒: ', num2str(time_abs, '%.3f')]};
    if isprop(h,'DisplayName') && ~isempty(get(h,'DisplayName'))
        txt{end+1} = ['数据源: ', get(h,'DisplayName')];
    end
    txt{end+1} = '点击起始点/结束点按钮进行标注';
end

function setStartPoint(~, ~)
    global data_storage;
    if isempty(data_storage.last_cursor_point)
        msgbox('请先使用数据游标选择一个数据点！', '提示', 'warn');
        return;
    end
    data_storage.current_start_point = data_storage.last_cursor_point;
    start_time = datestr(data_storage.current_start_point.time_dt, 'yyyy-mm-dd HH:MM:SS');
    info_text = sprintf('起始点: %s\n', start_time);
    if ~isempty(data_storage.current_end_point)
        end_time = datestr(data_storage.current_end_point.time_dt, 'yyyy-mm-dd HH:MM:SS');
        info_text = [info_text, sprintf('结束点: %s\n区间已选择，可以进行标注', end_time)];
    else
        info_text = [info_text, '请选择结束点'];
    end
    set(data_storage.annotation_info, 'String', info_text);
    updateAnnotationDisplay();
end

function setEndPoint(~, ~)
    global data_storage;
    if isempty(data_storage.last_cursor_point)
        msgbox('请先使用数据游标选择一个数据点！', '提示', 'warn');
        return;
    end
    data_storage.current_end_point = data_storage.last_cursor_point;
    end_time = datestr(data_storage.current_end_point.time_dt, 'yyyy-mm-dd HH:MM:SS');
    info_text = sprintf('结束点: %s\n', end_time);
    if ~isempty(data_storage.current_start_point)
        start_time = datestr(data_storage.current_start_point.time_dt, 'yyyy-mm-dd HH:MM:SS');
        info_text = [sprintf('起始点: %s\n', start_time), info_text, '区间已选择，可以进行标注'];
    else
        info_text = [info_text, '请选择起始点'];
    end
    set(data_storage.annotation_info, 'String', info_text);
    updateAnnotationDisplay();
end


function annotateRegion(label)
    global data_storage;
    if isempty(data_storage.current_start_point) || isempty(data_storage.current_end_point)
        msgbox('请先选择起始点和结束点！', '提示', 'warn');
        return;
    end
    
    % ---- 用绝对时间戳排序并存储 ----
    start_abs = min(data_storage.current_start_point.time_abs, data_storage.current_end_point.time_abs);
    end_abs   = max(data_storage.current_start_point.time_abs, data_storage.current_end_point.time_abs);
    % 同步保存可视化用的 datetime（仅展示，不用于比较）
    start_dt  = datetime(start_abs, 'ConvertFrom','posixtime', 'TimeZone','UTC');
    end_dt    = datetime(end_abs,   'ConvertFrom','posixtime', 'TimeZone','UTC');
    
    start_target = data_storage.current_start_point.target;
    data_source = '';
    if isprop(start_target, 'DisplayName') && ~isempty(get(start_target, 'DisplayName'))
        data_source = get(start_target, 'DisplayName');
    end
    
    annotation = struct();
    annotation.start_abs = start_abs;
    annotation.end_abs   = end_abs;
    annotation.start_dt  = start_dt;
    annotation.end_dt    = end_dt;
    annotation.label     = label;
    annotation.data_source = data_source;
    annotation.timestamp = now;
    
    data_storage.annotations{end+1} = annotation;
    
    msgbox(sprintf('标注完成！\n时间区间: %s 到 %s\n标签: %d\n数据源: %s', ...
                  datestr(start_dt, 'yyyy-mm-dd HH:MM:SS'), ...
                  datestr(end_dt,   'yyyy-mm-dd HH:MM:SS'), ...
                  label, data_source), '标注成功', 'help');
    
    data_storage.current_start_point = [];
    data_storage.current_end_point   = [];
    set(data_storage.annotation_info, 'String', sprintf('已添加标注 %d 个\n请继续选择新的时间区间', length(data_storage.annotations)));
    updateAnnotationDisplay();
end

function updateAnnotationDisplay()
    global data_storage;
    if ~get(data_storage.annotation_display_checkbox, 'Value')
        return;
    end
    hold(data_storage.axes, 'on');
    
    % 删除之前的标注显示
    children = get(data_storage.axes, 'Children');
    for i = 1:length(children)
        if isprop(children(i), 'Tag') && strcmp(get(children(i), 'Tag'), 'annotation_marker')
            delete(children(i));
        end
    end
    
    % 当前选择点（只显示，不比较）
    if ~isempty(data_storage.current_start_point)
        plot(data_storage.axes, data_storage.current_start_point.time_dt, data_storage.current_start_point.value, ...
             'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'green', 'Tag', 'annotation_marker');
    end
    if ~isempty(data_storage.current_end_point)
        plot(data_storage.axes, data_storage.current_end_point.time_dt, data_storage.current_end_point.value, ...
             'rv', 'MarkerSize', 12, 'MarkerFaceColor', 'red', 'Tag', 'annotation_marker');
    end
    
    % 已有标注区域（用 datetime 画，内部比较用 abs 秒）
    ylims = get(data_storage.axes, 'YLim');
    for i = 1:length(data_storage.annotations)
        ann = data_storage.annotations{i};
        if ann.label == 1, color = [0.2, 0.8, 0.2, 0.3];
        else,              color = [0.8, 0.2, 0.2, 0.3];
        end
        x_rect = [ann.start_dt, ann.end_dt, ann.end_dt, ann.start_dt];
        y_rect = [ylims(1), ylims(1), ylims(2), ylims(2)];
        fill(data_storage.axes, x_rect, y_rect, color, 'EdgeColor', 'none', 'Tag', 'annotation_marker');
    end
    hold(data_storage.axes, 'off');
end

function toggleAnnotationDisplay(~, ~)
    updateAnnotationDisplay();
end

function selectOutputFolder(~, ~)
    global data_storage;
    
    folder_path = uigetdir('', '选择输出文件夹');
    if folder_path ~= 0
        data_storage.output_folder = folder_path;
        set(data_storage.output_path_text, 'String', ['输出路径: ', folder_path]);
    end
end

function exportAnnotatedData(~, ~)
    global data_storage;
    
    if isempty(data_storage.data)
        msgbox('请先加载数据！', '警告', 'warn');
        return;
    end
    if isempty(data_storage.output_folder)
        msgbox('请先选择输出文件夹！', '警告', 'warn');
        return;
    end
    
    h_wait = waitbar(0, '正在导出标注数据...', 'Name', '导出进度');
    try
        for i = 1:length(data_storage.data)
            waitbar(i/length(data_storage.data), h_wait, ...
                   sprintf('处理文件 %d/%d: %s', i, length(data_storage.data), data_storage.data{i}.filename));
            
            file_info  = data_storage.data{i};
            data_table = file_info.data;

            % 确保有 t_abs（绝对秒）
            if ~ismember('t_abs', data_table.Properties.VariableNames)
                if ~isdatetime(data_table.time_utc)
                    error('导出前缺少 t_abs 且 time_utc 不是 datetime，无法生成绝对时间戳。');
                end
                data_table.t_abs = posixtime(data_table.time_utc);
            end
            
            % 默认 0
            label_column = zeros(height(data_table), 1);
            
            % 应用标注（仅用绝对时间戳比较）
            for j = 1:length(data_storage.annotations)
                ann = data_storage.annotations{j};
                % 若限定数据源，则过滤
                expected_source = sprintf('%s-%s', file_info.station, file_info.satellite);
                % if ~isempty(ann.data_source) && ~strcmp(ann.data_source, expected_source)
                %     continue;
                % end
                % 用绝对秒比较
                time_mask = data_table.t_abs >= ann.start_abs & data_table.t_abs <= ann.end_abs;
                label_column(time_mask) = ann.label;
            end
            
            data_table.label = label_column;
            output_filename = fullfile(data_storage.output_folder, [file_info.filename, '.csv']);
            writetable(data_table, output_filename);
        end
        close(h_wait);
        msgbox(sprintf('导出完成！\n已处理 %d 个文件\n输出路径: %s', ...
                      length(data_storage.data), data_storage.output_folder), '导出成功', 'help');
    catch ME
        if ishandle(h_wait), close(h_wait); end
        msgbox(['导出出错: ', ME.message], '错误', 'error');
    end
end

function clearAllAnnotations(~, ~)
    global data_storage;
    
    answer = questdlg('确定要清除所有标注吗？', '确认清除', '是', '否', '否');
    if strcmp(answer, '是')
        data_storage.annotations = {};
        data_storage.current_start_point = [];
        data_storage.current_end_point = [];
        set(data_storage.annotation_info, 'String', '所有标注已清除');
        updateAnnotationDisplay();
    end
end

function toggleDataCursor(src, ~)
    global data_storage;
    if isempty(data_storage.fig) || ~ishandle(data_storage.fig)
        data_storage.fig = ancestor(data_storage.axes, 'figure');
    end
    if get(src, 'Value')
        datacursormode(data_storage.fig, 'on');
    else
        datacursormode(data_storage.fig, 'off');
    end
end


function selectDataFolder(~, ~)
    global data_storage;
    
    % 选择数据文件夹
    folder_path = uigetdir('', '选择包含GPS数据文件的文件夹');
    if folder_path == 0
        return;
    end
    
    % 存储文件夹路径
    data_storage.folder_path = folder_path;
    
    % 查找所有CSV文件
    csv_files = dir(fullfile(folder_path, '*.csv'));
    
    if isempty(csv_files)
        msgbox('所选文件夹中没有找到CSV文件！', '警告', 'warn');
        return;
    end
    
    % 解析文件名并分类
    stations = {};
    satellites = {};
    doys = {};
    valid_files = {};
    
    for i = 1:length(csv_files)
        filename = csv_files(i).name;
        
        % 解析文件名格式：station_satellite_yearDOY.csv
        if length(filename) >= 16 && strcmp(filename(end-3:end), '.csv')
            parts = split(filename, '_');
            if length(parts) >= 3
                station = parts{1};
                satellite = parts{2};
                year_doy = parts{3}(1:end-4); % 去掉.csv
                
                if length(year_doy) == 7 % 2024153格式
                    year = year_doy(1:4);
                    doy = year_doy(5:7);
                    
                    stations{end+1} = station;
                    satellites{end+1} = satellite;
                    doys{end+1} = [year, '-', doy];
                    valid_files{end+1} = fullfile(folder_path, filename);
                end
            end
        end
    end
    
    if isempty(valid_files)
        msgbox('没有找到符合命名规范的文件！文件名格式应为：测站_卫星_年DOY.csv', '警告', 'warn');
        return;
    end
    
    % 存储文件信息
    data_storage.files = valid_files;
    data_storage.stations = unique(stations);
    data_storage.satellites = unique(satellites);
    data_storage.doys = unique(doys);
    
    % 排序显示
    data_storage.stations = sort(data_storage.stations);
    data_storage.satellites = sort(data_storage.satellites);
    data_storage.doys = sort(data_storage.doys);
    
    % 更新UI列表
    set(data_storage.station_listbox, 'String', data_storage.stations);
    set(data_storage.satellite_listbox, 'String', data_storage.satellites);
    set(data_storage.doy_listbox, 'String', data_storage.doys);
    
    % 更新状态
    set(data_storage.status_text, 'String', sprintf('已扫描：%d个文件', length(valid_files)));
    set(data_storage.scan_info, 'String', sprintf('测站: %d 个 | 卫星: %d 个 | DOY: %d 个', ...
                        length(data_storage.stations), length(data_storage.satellites), length(data_storage.doys)));
    
    % 清空之前加载的数据
    data_storage.data = {};
    set(data_storage.load_status_text, 'String', '请选择要加载的数据');
end

function loadSelectedFiles(~, ~)
    global data_storage;
    
    if isempty(data_storage.files)
        msgbox('请先选择数据文件夹！', '警告', 'warn');
        return;
    end
    
    % 获取用户选择
    selected_stations_idx = get(data_storage.station_listbox, 'Value');
    selected_satellites_idx = get(data_storage.satellite_listbox, 'Value');
    selected_doys_idx = get(data_storage.doy_listbox, 'Value');
    
    if isempty(selected_stations_idx) || isempty(selected_satellites_idx) || isempty(selected_doys_idx)
        msgbox('请先选择要加载的测站、卫星和DOY！', '警告', 'warn');
        return;
    end
    
    selected_stations = data_storage.stations(selected_stations_idx);
    selected_satellites = data_storage.satellites(selected_satellites_idx);
    selected_doys = data_storage.doys(selected_doys_idx);
    
    % 过滤需要加载的文件
    files_to_load = {};
    for i = 1:length(data_storage.files)
        filename = data_storage.files{i};
        [~, base_name, ~] = fileparts(filename);
        parts = split(base_name, '_');
        
        if length(parts) >= 3
            station = parts{1};
            satellite = parts{2};
            year_doy = parts{3};
            year = year_doy(1:4);
            doy = year_doy(5:7);
            formatted_doy = [year, '-', doy];
            
            % 检查是否在选择列表中
            if any(strcmp(selected_stations, station)) && ...
               any(strcmp(selected_satellites, satellite)) && ...
               any(strcmp(selected_doys, formatted_doy))
                files_to_load{end+1} = filename;
            end
        end
    end
    
    if isempty(files_to_load)
        msgbox('没有找到符合选择条件的文件！', '警告', 'warn');
        return;
    end
    
    % 显示进度条
    h_wait = waitbar(0, '正在加载选中的数据文件...', 'Name', '数据加载进度');
    
    data_storage.data = {};
    
    try
        for i = 1:length(files_to_load)
            waitbar(i/length(files_to_load), h_wait, ...
                   sprintf('加载文件 %d/%d: %s', i, length(files_to_load), ...
                          basename(files_to_load{i})));
            
            filename = files_to_load{i};
            
            % 读取CSV文件
            try
                data = readtable(filename);

                % ---- 统一成 UTC datetime + 绝对时间戳（秒）----
                if iscell(data.time_utc)
                    data.time_utc = datetime(data.time_utc, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS', 'TimeZone','UTC');
                elseif ischar(data.time_utc) || isstring(data.time_utc)
                    data.time_utc = datetime(data.time_utc, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS', 'TimeZone','UTC');
                elseif isdatetime(data.time_utc)
                    if isempty(data.time_utc.TimeZone) || strcmp(data.time_utc.TimeZone,'')
                        data.time_utc.TimeZone = 'UTC';
                    else
                        data.time_utc = datetime(data.time_utc, 'TimeZone','UTC');
                    end
                else
                    error('time_utc 列类型不支持，请确认为字符串或 datetime。');
                end

                % 绝对时间戳（秒，double）
                if ~ismember('t_abs', data.Properties.VariableNames)
                    data.t_abs = posixtime(data.time_utc);
                else
                    % 若已存在，确保是 double
                    data.t_abs = double(data.t_abs);
                end
                % ----------------------------------------------

                % 存储数据和文件信息
                [~, base_name, ~] = fileparts(filename);
                parts = split(base_name, '_');
                
                file_info = struct();
                file_info.data = data;
                file_info.station = parts{1};
                file_info.satellite = parts{2};
                file_info.year_doy = parts{3};
                file_info.filename = base_name;
                
                data_storage.data{end+1} = file_info;
                
            catch ME
                warning('读取文件失败: %s, 错误: %s', filename, ME.message);
            end
        end
        
        close(h_wait);
        
        % 更新状态
        set(data_storage.load_status_text, 'String', sprintf('已加载: %d个文件', length(data_storage.data)));
        
        msgbox(sprintf('数据加载完成！\n成功加载 %d 个文件\n可以开始数据分析和标注', length(data_storage.data)), '加载完成', 'help');
        
    catch ME
        if ishandle(h_wait)
            close(h_wait);
        end
        msgbox(['数据加载出错: ', ME.message], '错误', 'error');
    end
end

function clearLoadedData(~, ~)
    global data_storage;
    
    answer = questdlg('确定要清空已加载的数据吗？', '确认清空', '是', '否', '否');
    if strcmp(answer, '是')
        data_storage.data = {};
        data_storage.annotations = {};
        data_storage.current_start_point = [];
        data_storage.current_end_point = [];
        set(data_storage.load_status_text, 'String', '已清空数据');
        set(data_storage.annotation_info, 'String', '请先用数据游标选择点，然后设置起始点和结束点');
        clearPlots();
    end
end

function name = basename(filepath)
    [~, name, ext] = fileparts(filepath);
    name = [name, ext];
end

function selectAllItems(listbox)
    items = get(listbox, 'String');
    if ~isempty(items)
        set(listbox, 'Value', 1:length(items));
    end
end

function plotTimeSeries(~, ~)
    global data_storage;
    
    if isempty(data_storage.data)
        msgbox('请先加载数据！', '警告', 'warn');
        return;
    end
    
    param_idx = get(data_storage.param_dropdown, 'Value');
    param_names = get(data_storage.param_dropdown, 'String');
    selected_param = param_names{param_idx};
    
    cla(data_storage.axes);
    hold(data_storage.axes, 'on');
    
    num_series = length(data_storage.data);
    colors = lines(num_series);
    legend_entries = {};
    data_values = [];
    total_points = 0;
    
    try
        for i = 1:length(data_storage.data)
            file_info  = data_storage.data{i};
            data_table = file_info.data;

            % 确保 time_utc 是 UTC datetime；确保 t_abs 存在
            if ~isdatetime(data_table.time_utc)
                error('数据 time_utc 不是 datetime，请检查加载逻辑。');
            end
            if isempty(data_table.time_utc.TimeZone)
                data_table.time_utc.TimeZone = 'UTC';
            end
            if ~ismember('t_abs', data_table.Properties.VariableNames)
                data_table.t_abs = posixtime(data_table.time_utc);
            end
            
            if any(strcmp(data_table.Properties.VariableNames, selected_param))
                times  = data_table.time_utc;
                values = data_table.(selected_param);
                
                valid_idx = ~isnan(values) & isfinite(values);
                times_plot  = times(valid_idx);
                values_plot = values(valid_idx);
                row_idx     = find(valid_idx);  % <- 保留“原表行号”映射
                
                if ~isempty(times_plot)
                    h = scatter(data_storage.axes, times_plot, values_plot, 36, colors(i, :), ...
                                'filled', 'o', 'DisplayName', sprintf('%s-%s', file_info.station, file_info.satellite));
                    
                    % 绑定来源：这个散点对应哪个 file，以及对应原表中的哪些行
                    setappdata(h, 'file_idx', i);
                    setappdata(h, 'row_idx',  row_idx);
                    
                    legend_entries{end+1} = sprintf('%s-%s (%s)', file_info.station, file_info.satellite, file_info.year_doy);
                    total_points = total_points + numel(values_plot);
                    data_values = [data_values; values_plot];
                end
            end
        end
        
        xlabel(data_storage.axes, '时间 (UTC)', 'FontSize', 12);
        ylabel(data_storage.axes, selected_param, 'FontSize', 12);
        title(data_storage.axes, sprintf('%s 时间序列分析', selected_param), 'FontSize', 14);
        
        if get(data_storage.grid_checkbox, 'Value'), grid(data_storage.axes, 'on'); else, grid(data_storage.axes, 'off'); end
        if get(data_storage.legend_checkbox, 'Value') && ~isempty(legend_entries)
            legend(data_storage.axes, legend_entries, 'Location', 'best', 'FontSize', 9);
        end
        
        if get(data_storage.cursor_checkbox, 'Value')
            if isempty(data_storage.fig) || ~ishandle(data_storage.fig)
                data_storage.fig = ancestor(data_storage.axes, 'figure');
            end
            datacursormode(data_storage.fig, 'on');
        end
        
        if ~isempty(data_values)
            stats_text = sprintf('数据点数: %d | 最小值: %.3f | 最大值: %.3f\n平均值: %.3f | 标准差: %.3f\n标注区域: %d 个', ...
                               total_points, min(data_values), max(data_values), ...
                               mean(data_values), std(data_values), length(data_storage.annotations));
        else
            stats_text = '没有有效数据';
        end
        set(data_storage.stats_text, 'String', stats_text);
        
        hold(data_storage.axes, 'off');
        updateAnnotationDisplay();
        axis(data_storage.axes, 'tight');
        
    catch ME
        msgbox(['绘图出错: ', ME.message], '错误', 'error');
        fprintf('详细错误信息: %s\n', getReport(ME));
    end
end

function clearPlots(~, ~)
    global data_storage;
    cla(data_storage.axes);
    title(data_storage.axes, '请选择数据并点击绘制时间序列', 'FontSize', 14);
    xlabel(data_storage.axes, '时间 (UTC)', 'FontSize', 12);
    ylabel(data_storage.axes, '数值', 'FontSize', 12);
    grid(data_storage.axes, 'on');
    set(data_storage.stats_text, 'String', '统计信息将在此显示');
    
    % 清除标注选择
    data_storage.current_start_point = [];
    data_storage.current_end_point = [];
    set(data_storage.annotation_info, 'String', '请先用数据游标选择点，然后设置起始点和结束点');
end