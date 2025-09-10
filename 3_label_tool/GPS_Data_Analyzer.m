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
    fig = figure('Name', 'GPS数据时序分析器', 'Position', [50, 100, 1600, 900], ...
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
    % 现在把“数据加载”放在最底部
    row_gap      = 0.02;
    load_area_h  = 0.20;   % 加载按钮区域高度（底部）
    param_area_h = 0.08;   % 参数选择区域高度（靠近底部上方）
    select_area_h = (1 - load_area_h - param_area_h - 5*row_gap) / 3; % 三个选择框均分（顶部到中部）

    % 从上往下依次：测站 -> 卫星 -> DOY -> 参数 -> 加载(底部)
    y_station   = 1 - row_gap - select_area_h;
    y_satellite = y_station  - row_gap - select_area_h;
    y_doy       = y_satellite - row_gap - select_area_h;
    y_param     = y_doy      - row_gap - param_area_h;
    y_load      = row_gap;  % 最底部

    % —— 测站选择 —— %
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

    % —— 卫星选择 —— %
    satellite_panel = uipanel('Parent', select_panel, 'Units','normalized', ...
        'Position', [0.02, y_satellite, 0.96, select_area_h], 'Title', '卫星选择', ...
        'FontWeight', 'bold', 'FontSize', 9, 'BackgroundColor', [0.98, 0.98, 1]);
    
    data_storage.satellite_listbox = uicontrol('Parent', satellite_panel, 'Style', 'listbox', ...
        'Units','normalized','Position', [0.03, 0.25, 0.82, 0.65], 'Max', 50, ...
        'String', {}, 'Value', [], 'FontSize', 8);
    
    uicontrol('Parent', satellite_panel, 'Style', 'pushbutton', 'String', '全选', ...
        'Units','normalized','Position', [0.87, 0.58, 0.10, 0.30], 'FontSize', 8, ...
        'Callback', @(~,~) selectAllItems(data_storage.satellite_listbox));
    uicontrol('Parent', satellite_panel, 'Style', 'pushbutton', 'String', '清空', ...
        'Units','normalized','Position', [0.87, 0.18, 0.10, 0.30], 'FontSize', 8, ...
        'Callback', @(~,~) set(data_storage.satellite_listbox, 'Value', []));
    
    % 添加复选框：是否仅显示FP卫星
    data_storage.fp_only_checkbox = uicontrol('Parent', satellite_panel, 'Style', 'checkbox', ...
        'String', '仅显示FP卫星', 'Units','normalized','Position', [0.03, 0.02, 0.40, 0.20], ...
        'Value', 0, 'FontSize', 8, 'BackgroundColor', [0.98, 0.98, 1], ...
        'Callback', @toggleFPSatellites);

    % —— DOY 选择 —— %
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

    % —— 参数选择（位于加载区上方） —— %
    param_panel = uipanel('Parent', select_panel, 'Units','normalized', ...
        'Position', [0.02, y_param, 0.96, param_area_h], 'BorderType', 'none', ...
        'BackgroundColor', [0.98, 1, 0.98]);
    uicontrol('Parent', param_panel, 'Style', 'text', 'String', '绘图参数:', ...
        'Units','normalized', 'Position', [0.02, 0.30, 0.20, 0.60], 'FontWeight', 'bold', ...
        'FontSize', 9, 'BackgroundColor', [0.98, 1, 0.98], 'HorizontalAlignment','left');

    data_storage.param_dropdown = uicontrol('Parent', param_panel, 'Style', 'popupmenu', ...
        'String', {'S1C', 'S2W', 'S2W_S1C_diff', 'elevation', 'azimuth', 'slant_range'}, ...
        'Units','normalized','Position', [0.22, 0.30, 0.35, 0.60], 'FontSize', 9);

    % —— 数据加载（最底部） —— %
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

    % ====== 数据标注区 ======
    annotation_panel = uipanel('Parent', control_panel, 'Units','normalized', ...
        'Position', [left, y_annot, width, h_annot], ...
        'Title', '数据标注', 'FontWeight', 'bold', 'Backgr                                oundColor', [1, 0.95, 0.9]);

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
    % 在 annotation_panel 增加“当天全标为 1”按钮
    uicontrol('Parent', annotation_panel, 'Style', 'pushbutton', 'String', '当天全标为 1', ...
        'Units','normalized','Position', [0.03, 0.35, 0.30, 0.2], 'FontSize', 9, ...
        'BackgroundColor', [0.1, 0.5, 0.9], 'ForegroundColor', 'white', ...
        'Callback', @markWholeDayOne);

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
    
    % 数据游标：绑定 UpdateFcn，并关闭“吸附到数据点”
    data_storage.dcm = datacursormode(data_storage.fig);
    set(data_storage.dcm, 'UpdateFcn', @dataCursorUpdateFcn);
    
    % 关键：关闭吸附（不同版本 MATLAB 属性名可能略有差异）
    try
        set(data_storage.dcm, 'SnapToDataVertex', 'off');  % 老版 datacursormode
    catch
    end
    
    datacursormode(data_storage.fig, 'on');

    % 其它初始化
    data_storage.output_folder = '';
    data_storage.last_cursor_point = [];
    data_storage.folder_path = '';  % 添加存储文件夹路径
end


function txt = dataCursorUpdateFcn(~, event_obj)
    global data_storage;
    h   = get(event_obj, 'Target');      % 当前 DataTip 作用的散点
    pos = get(event_obj, 'Position');    % [x, y]，x 可能是 datenum/小数天 或 datetime

    % 取该散点预先绑定的 DOY 基准日期（UTC日零点）
    base_date = getappdata(h, 'base_date');
    if isempty(base_date)
        % 兜底：从本系列的 x_plot 推断（通常不会走到）
        x_plot = getappdata(h, 'x_plot');
        if isdatetime(x_plot)
            if isempty(x_plot.TimeZone), x_plot.TimeZone = 'UTC'; end
            base_date = dateshift(min(x_plot), 'start', 'day');
        else
            base_date = datetime(1970,1,1,'TimeZone','UTC');
        end
    end

    % 从 DataTip 的横坐标提取“当日时分秒”
    xq = pos(1);
    if isdatetime(xq)
        % 若 DataTip 给的是 datetime，直接取其中的“timeofday”
        if isempty(xq.TimeZone), xq.TimeZone = 'UTC'; else, xq = datetime(xq,'TimeZone','UTC'); end
        tod = timeofday(xq);                % duration
    else
        % 多数情况下 xq 是“小数天”（如 0.229xx 表示当天 05:30:xx）
        % 用小数天转为 duration（时分秒）；即使 xq 是完整 datenum，也只取小数部分做当日时刻
        tod = days(rem(double(xq), 1));     % 只保留当日的小数部分 -> duration
    end

    % 拼成“正确的绝对时间”（UTC）
    time_dt  = base_date + tod;             % base_date 是当日 00:00:00
    time_abs = posixtime(time_dt);          % 绝对秒（后续比较/导出仍用它）

    % y 值：就用 DataTip 的 y（不吸附）
    y_val = double(pos(2));

    % 缓存“最后一次 DataTip 选中”的点，供起止点按钮取用
    data_storage.last_cursor_point = struct( ...
        'time_dt',  time_dt, ...
        'time_abs', double(time_abs), ...
        'value',    y_val, ...
        'target',   h);

    % DataTip 显示文字（可按需精简）
    % 注意：这里只显示到秒；若你有毫秒，可以把格式改成 'yyyy-mm-dd HH:MM:SS.FFF'
    txt = { ...
        ['时间: ', datestr(time_dt, 'yyyy-mm-dd HH:MM:SS')], ...
        ['值: ', num2str(y_val, '%.3f')], ...
        ['绝对秒: ', num2str(time_abs, '%.3f')], ...
        '（此时间即用于设为起始/结束）' ...
    };
end

function setStartPoint(~, ~)
    global data_storage;
    if isempty(data_storage.last_cursor_point)
        % 不弹窗，只在面板提示
        set(data_storage.annotation_info,'String','请先在图上用数据提示选择一个时间点（拖动后按 Enter 最稳）。');
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
        set(data_storage.annotation_info,'String','请先在图上用数据提示选择一个时间点。');
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
        set(data_storage.annotation_info,'String','请先设置起始点和结束点，再进行标注。');
        return;
    end

    start_abs = min(data_storage.current_start_point.time_abs, data_storage.current_end_point.time_abs);
    end_abs   = max(data_storage.current_start_point.time_abs, data_storage.current_end_point.time_abs);
    start_dt  = datetime(start_abs,'ConvertFrom','posixtime','TimeZone','UTC');
    end_dt    = datetime(end_abs,  'ConvertFrom','posixtime','TimeZone','UTC');

    start_target = data_storage.current_start_point.target;
    data_source = '';
    if isprop(start_target,'DisplayName') && ~isempty(get(start_target,'DisplayName'))
        data_source = get(start_target,'DisplayName');
    end

    annotation = struct('start_abs',start_abs,'end_abs',end_abs, ...
                        'start_dt',start_dt,'end_dt',end_dt, ...
                        'label',label,'data_source',data_source,'timestamp',now);
    data_storage.annotations{end+1} = annotation;

    % 面板提示替代弹窗
    set(data_storage.annotation_info,'String',sprintf( ...
        '标注完成：[%s ~ %s]，label=%d，源=%s\n已添加标注 %d 个。', ...
        datestr(start_dt,'yyyy-mm-dd HH:MM:SS'), ...
        datestr(end_dt,'yyyy-mm-dd HH:MM:SS'), ...
        label, data_source, numel(data_storage.annotations)));

    data_storage.current_start_point = [];
    data_storage.current_end_point   = [];
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
    
    folder_path = uigetdir('E:\projects\ML_FP\data\3_labeled_raw_datasets\2024', '选择输出文件夹');
    if folder_path ~= 0
        data_storage.output_folder = folder_path;
        set(data_storage.output_path_text, 'String', ['输出路径: ', folder_path]);
    end
end

function exportAnnotatedData(~, ~)
    global data_storage;

    if isempty(data_storage.data)
        set(data_storage.annotation_info,'String','未加载数据，无法导出。');
        return;
    end
    if isempty(data_storage.output_folder)
        set(data_storage.annotation_info,'String','未设置输出路径，请先选择输出文件夹。');
        return;
    end

    written = 0;
    for i = 1:length(data_storage.data)
        file_info  = data_storage.data{i};
        data_table = file_info.data;

        if ~ismember('t_abs', data_table.Properties.VariableNames)
            if ~isdatetime(data_table.time_utc)
                set(data_storage.annotation_info,'String','导出失败：time_utc 非 datetime，无法生成 t_abs。');
                return;
            end
            data_table.t_abs = posixtime(data_table.time_utc);
        end

        label_column = zeros(height(data_table),1);
        for j = 1:length(data_storage.annotations)
            ann = data_storage.annotations{j};
            time_mask = data_table.t_abs >= ann.start_abs & data_table.t_abs <= ann.end_abs;
            label_column(time_mask) = ann.label;
        end
        data_table.label = label_column;

        output_filename = fullfile(data_storage.output_folder, [file_info.filename, '.csv']);
        try
            writetable(data_table, output_filename);
            written = written + 1;
        catch ME
            fprintf('导出失败: %s, 错误: %s\n', output_filename, ME.message);
        end
        set(data_storage.annotation_info,'String',sprintf('导出进度：%d/%d -> %s', i, length(data_storage.data), file_info.filename));
        drawnow;
    end

    set(data_storage.annotation_info,'String',sprintf('导出完成。成功导出 %d/%d 个文件。输出目录：%s', ...
        written, length(data_storage.data), data_storage.output_folder));
end

function clearAllAnnotations(~, ~)
    global data_storage;
    data_storage.annotations = {};
    data_storage.current_start_point = [];
    data_storage.current_end_point   = [];
    set(data_storage.annotation_info,'String','所有标注已清除。');
    updateAnnotationDisplay();
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

    folder_path = uigetdir('E:\projects\ML_FP\data\2_raw_datasets\2024', '选择包含GPS数据文件的文件夹');
    if folder_path == 0
        set(data_storage.status_text,'String','未选择文件夹。');
        return;
    end
    data_storage.folder_path = folder_path;

    csv_files = dir(fullfile(folder_path, '*.csv'));
    if isempty(csv_files)
        set(data_storage.status_text,'String','所选文件夹中没有找到CSV文件。');
        set(data_storage.scan_info,'String','测站: 0 | 卫星: 0 | DOY: 0');
        set(data_storage.station_listbox,'String',{}); set(data_storage.satellite_listbox,'String',{}); set(data_storage.doy_listbox,'String',{});
        return;
    end

    stations = {}; satellites = {}; doys = {}; valid_files = {};
    for i = 1:length(csv_files)
        filename = csv_files(i).name;
        if length(filename) >= 16 && strcmp(filename(end-3:end), '.csv')
            parts = split(filename, '_');
            if length(parts) >= 3
                station = parts{1};
                satellite = parts{2};
                year_doy = parts{3}(1:end-4);
                if length(year_doy) == 7
                    year = year_doy(1:4); doy = year_doy(5:7);
                    stations{end+1} = station; %#ok<AGROW>
                    satellites{end+1} = satellite; %#ok<AGROW>
                    doys{end+1} = [year, '-', doy]; %#ok<AGROW>
                    valid_files{end+1} = fullfile(folder_path, filename); %#ok<AGROW>
                end
            end
        end
    end

    if isempty(valid_files)
        set(data_storage.status_text,'String','未找到符合命名规范的文件（测站_卫星_年DOY.csv）。');
        set(data_storage.scan_info,'String','测站: 0 | 卫星: 0 | DOY: 0');
        return;
    end

    data_storage.files = valid_files;
    data_storage.stations = sort(unique(stations));
    data_storage.satellites = sort(unique(satellites));
    data_storage.doys = sort(unique(doys));

    set(data_storage.station_listbox,'String',data_storage.stations);
    set(data_storage.satellite_listbox,'String',data_storage.satellites);
    set(data_storage.doy_listbox,'String',data_storage.doys);

    % 新增：如果勾选了“仅显示FP卫星”，立即刷新
    if isfield(data_storage,'fp_only_checkbox') && get(data_storage.fp_only_checkbox,'Value')
        toggleFPSatellites(data_storage.fp_only_checkbox);
    end

    set(data_storage.status_text,'String',sprintf('已扫描：%d 个文件', length(valid_files)));
    set(data_storage.scan_info,'String',sprintf('测站: %d | 卫星: %d | DOY: %d', ...
        length(data_storage.stations), length(data_storage.satellites), length(data_storage.doys)));

    data_storage.data = {};
    set(data_storage.load_status_text,'String','请选择要加载的数据');
end

function loadSelectedFiles(~, ~)
    global data_storage;

    if isempty(data_storage.files)
        set(data_storage.load_status_text,'String','未选择数据文件夹。');
        return;
    end

    sel_st = get(data_storage.station_listbox,'Value');
    sel_sa = get(data_storage.satellite_listbox,'Value');
    sel_do = get(data_storage.doy_listbox,'Value');
    if isempty(sel_st) || isempty(sel_sa) || isempty(sel_do)
        set(data_storage.load_status_text,'String','请先选择测站、卫星和DOY。');
        return;
    end

    selected_stations  = data_storage.stations(sel_st);
    selected_satellites= data_storage.satellites(sel_sa);
    selected_doys      = data_storage.doys(sel_do);

    files_to_load = {};
    for i = 1:length(data_storage.files)
        filename = data_storage.files{i};
        [~, base_name, ~] = fileparts(filename);
        parts = split(base_name, '_');
        if length(parts) >= 3
            station = parts{1}; satellite = parts{2}; year_doy = parts{3};
            formatted_doy = [year_doy(1:4), '-', year_doy(5:7)];
            if any(strcmp(selected_stations, station)) && any(strcmp(selected_satellites, satellite)) && any(strcmp(selected_doys, formatted_doy))
                files_to_load{end+1} = filename; %#ok<AGROW>
            end
        end
    end

    if isempty(files_to_load)
        set(data_storage.load_status_text,'String','没有符合选择条件的文件。');
        return;
    end

    data_storage.data = {};
    set(data_storage.load_status_text,'String',sprintf('开始加载 %d 个文件...', length(files_to_load))); drawnow;

    success = 0;
    for i = 1:length(files_to_load)
        filename = files_to_load{i};
        set(data_storage.load_status_text,'String',sprintf('加载中 %d/%d: %s', i, length(files_to_load), basename(filename))); drawnow;
        try
            data = readtable(filename);

            if iscell(data.time_utc)
                data.time_utc = datetime(data.time_utc,'InputFormat','yyyy-MM-dd HH:mm:ss.SSS','TimeZone','UTC');
            elseif ischar(data.time_utc) || isstring(data.time_utc)
                data.time_utc = datetime(data.time_utc,'InputFormat','yyyy-MM-dd HH:mm:ss.SSS','TimeZone','UTC');
            elseif isdatetime(data.time_utc)
                if isempty(data.time_utc.TimeZone) || strcmp(data.time_utc.TimeZone,'')
                    data.time_utc.TimeZone = 'UTC';
                else
                    data.time_utc = datetime(data.time_utc,'TimeZone','UTC');
                end
            else
                set(data_storage.load_status_text,'String','time_utc 列类型不支持，请确认为字符串或 datetime。'); return;
            end

            if ~ismember('t_abs', data.Properties.VariableNames)
                data.t_abs = posixtime(data.time_utc);
            else
                data.t_abs = double(data.t_abs);
            end

            [~, base_name, ~] = fileparts(filename);
            parts = split(base_name, '_');
            file_info = struct();
            file_info.data = data;
            file_info.station = parts{1};
            file_info.satellite = parts{2};
            file_info.year_doy = parts{3};
            file_info.filename = base_name;
            data_storage.data{end+1} = file_info; %#ok<AGROW>
            success = success + 1;
        catch ME
            % 控制台输出即可，不弹窗
            fprintf('读取文件失败: %s, 错误: %s\n', filename, ME.message);
        end
    end

    set(data_storage.load_status_text,'String',sprintf('数据加载完成。成功: %d / 总计: %d', success, length(files_to_load)));
end


function clearLoadedData(~, ~)
    global data_storage;
    data_storage.data = {};
    data_storage.annotations = {};
    data_storage.current_start_point = [];
    data_storage.current_end_point = [];
    set(data_storage.load_status_text,'String','已清空数据');
    set(data_storage.annotation_info,'String','请先用数据游标选择点，然后设置起始点和结束点');
    clearPlots();
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
        set(data_storage.stats_text,'String','未加载数据，无法绘图。');
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

            if ~isdatetime(data_table.time_utc), set(data_storage.stats_text,'String','time_utc 非 datetime。'); return; end
            if isempty(data_table.time_utc.TimeZone), data_table.time_utc.TimeZone = 'UTC'; end
            if ~ismember('t_abs', data_table.Properties.VariableNames), data_table.t_abs = posixtime(data_table.time_utc); end

            if any(strcmp(data_table.Properties.VariableNames, selected_param))
                times  = data_table.time_utc;
                values = data_table.(selected_param);
                valid_idx   = ~isnan(values) & isfinite(values);
                times_plot  = times(valid_idx);
                values_plot = values(valid_idx);
                row_idx     = find(valid_idx);

                if ~isempty(times_plot)
                    h = scatter(data_storage.axes, times_plot, values_plot, 36, colors(i,:), ...
                        'filled','o','DisplayName',sprintf('%s-%s',file_info.station,file_info.satellite), ...
                        'PickableParts','all','HitTest','on');

                    yd  = file_info.year_doy;
                    yr  = str2double(yd(1:4));
                    doy = str2double(yd(5:7));
                    base_date = datetime(yr,1,1,'TimeZone','UTC') + days(doy-1);
                    setappdata(h,'base_date', base_date);

                    setappdata(h,'file_idx',i);
                    setappdata(h,'row_idx', row_idx);
                    setappdata(h,'x_plot',  times_plot);
                    setappdata(h,'y_plot',  values_plot);

                    legend_entries{end+1} = sprintf('%s-%s (%s)', file_info.station, file_info.satellite, file_info.year_doy); %#ok<AGROW>
                    total_points = total_points + numel(values_plot);
                    data_values  = [data_values; values_plot]; %#ok<AGROW>
                end
            end
        end

        xlabel(data_storage.axes,'时间 (UTC)','FontSize',12);
        ylabel(data_storage.axes,selected_param,'FontSize',12);
        title(data_storage.axes,sprintf('%s 时间序列分析',selected_param),'FontSize',14);

        if get(data_storage.grid_checkbox,'Value'), grid(data_storage.axes,'on'); else, grid(data_storage.axes,'off'); end
        if get(data_storage.legend_checkbox,'Value') && ~isempty(legend_entries)
            legend(data_storage.axes, legend_entries, 'Location','best','FontSize',9);
        end

        if get(data_storage.cursor_checkbox,'Value')
            if isempty(data_storage.fig) || ~ishandle(data_storage.fig)
                data_storage.fig = ancestor(data_storage.axes,'figure');
            end
            datacursormode(data_storage.fig,'on');
        end

        if ~isempty(data_values)
            stats_text = sprintf('数据点数: %d | 最小值: %.3f | 最大值: %.3f\n平均值: %.3f | 标准差: %.3f | 标注: %d 个', ...
                total_points, min(data_values), max(data_values), mean(data_values), std(data_values), length(data_storage.annotations));
        else
            stats_text = '没有有效数据';
        end
        set(data_storage.stats_text,'String',stats_text);

        hold(data_storage.axes,'off');
        updateAnnotationDisplay();
        axis(data_storage.axes,'tight');
    catch ME
        set(data_storage.stats_text,'String',['绘图出错：', ME.message]);
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

function toggleFPSatellites(src, ~)
    global data_storage;
    fp_sats = { ...
        'G01','G03','G05','G06','G07','G08','G09','G10','G12','G15','G17', ...
        'G24','G25','G26','G27','G29','G30','G31','G32'};

    if get(src,'Value')  % 仅显示FP卫星
        filtered = intersect(data_storage.satellites, fp_sats, 'stable');
        set(data_storage.satellite_listbox,'String',filtered,'Value',[]);
    else                 % 显示全部卫星
        set(data_storage.satellite_listbox,'String',data_storage.satellites,'Value',[]);
    end
end

function onKeyPress(~, event)
    % Ctrl+1 快捷键：当天全标为 1
    if isfield(event,'Modifier') && any(strcmpi(event.Modifier,'control')) && strcmp(event.Key,'1')
        markWholeDayOne();
    end
end

function markWholeDayOne(~, ~)
    global data_storage;

    % 1) 优先使用最后一次数据游标所在系列的 base_date
    base_date = [];
    data_source = '';
    if ~isempty(data_storage.last_cursor_point) && isfield(data_storage.last_cursor_point,'target') ...
            && isgraphics(data_storage.last_cursor_point.target)
        h = data_storage.last_cursor_point.target;
        if ~isempty(getappdata(h,'base_date'))
            base_date = getappdata(h,'base_date');  % datetime(YYYY,1,1,'TimeZone','UTC') + days(doy-1)
            if isprop(h,'DisplayName') && ~isempty(get(h,'DisplayName'))
                data_source = get(h,'DisplayName');
            end
        end
    end

    % 2) 若没有数据游标，且只加载了一个 DOY，则用唯一 DOY 的日零点
    if isempty(base_date)
        if isempty(data_storage.data)
            set(data_storage.annotation_info,'String','未加载数据。请加载后再试。');
            return;
        end
        % 检查已加载文件的 DOY 是否唯一
        yd_all = cellfun(@(fi) fi.year_doy, data_storage.data, 'UniformOutput', false);
        yd_uniq = unique(yd_all);
        if numel(yd_uniq) == 1
            yd = yd_uniq{1};  % 'YYYYDOY'
            yr  = str2double(yd(1:4));
            doy = str2double(yd(5:7));
            base_date = datetime(yr,1,1,'TimeZone','UTC') + days(doy-1);
            data_source = sprintf('ALL(%s)', yd);
        else
            set(data_storage.annotation_info,'String','存在多个 DOY，且未选中数据点。请先在图上用数据游标选一个点来确定“当天”。');
            return;
        end
    end

    % 3) 计算整天范围 [00:00:00, 23:59:59.999]
    day_start_dt = dateshift(base_date,'start','day');      % 当天 00:00:00
    day_end_dt   = day_start_dt + days(1) - milliseconds(1);% 当天 23:59:59.999
    start_abs = posixtime(day_start_dt);
    end_abs   = posixtime(day_end_dt);

    % 4) 写入一条 label=1 的整天标注
    ann = struct('start_abs',start_abs,'end_abs',end_abs, ...
                 'start_dt',day_start_dt,'end_dt',day_end_dt, ...
                 'label',1,'data_source',data_source,'timestamp',now);
    data_storage.annotations{end+1} = ann;

    % 5) 提示 + 刷新
    set(data_storage.annotation_info,'String',sprintf( ...
        '当天全标完成：[%s ~ %s]，label=1，源=%s\n当前标注总数：%d', ...
        datestr(day_start_dt,'yyyy-mm-dd HH:MM:SS.FFF'), ...
        datestr(day_end_dt,  'yyyy-mm-dd HH:MM:SS.FFF'), ...
        data_source, numel(data_storage.annotations)));
    updateAnnotationDisplay();
end

