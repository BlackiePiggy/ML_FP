if ~isdatetime(data_table2.time_utc)
    data_table2.time_utc = datetime(data_table2.time_utc, 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
else
    % 如果已经是 datetime 类型，确保格式正确
    data_table2.time_utc.Format = 'yyyy-MM-dd HH:mm:ss.SSS';
end

% 获取列名（明确使用time_utc作为时间列）
colNames = data_table2.Properties.VariableNames;
% 找到时间列之后的两个数据列（根据movevars操作，应该是第2和第3列）
dataCol1 = colNames{2};  
dataCol2 = colNames{3};  

% 创建图形
figure('Name', '时间序列对比图', 'Position', [100 100 1000 600]);

% 绘制第一个数据列（使用time_utc作为x轴）
plot(data_table2.time_utc, data_table2.(dataCol1), 'b-', 'LineWidth', 1.2);
hold on;

% 绘制第二个数据列
plot(data_table2.time_utc, data_table2.(dataCol2), 'r--', 'LineWidth', 1.2);

% 设置时间轴格式
xtickformat('yyyy-MM-dd HH:mm');
xtickangle(45);  % 旋转标签防止重叠

% 添加图表元素
legend(dataCol1, dataCol2, 'Location', 'best');
title('时间序列对比图');
xlabel('时间');
ylabel('数值');
grid on;

hold off;