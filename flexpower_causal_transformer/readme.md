# Flex Power Detection System

基于深度学习的GPS Flex Power状态检测系统

## 项目概述

本项目实现了一个端到端的深度学习模型，用于检测GPS卫星的Flex Power状态。Flex Power是GPS卫星的一种功能，开启后会在S2W频段产生载噪比(CNR)的幅值抬升。本系统通过融合多模态数据和时序特征，实现了高精度、高泛化性的Flex Power状态检测。

## 主要特性

- **多模态数据融合**：综合利用S2W、S1C载噪比数据、空间位置信息和时序特征
- **Transformer-CNN混合架构**：结合CNN的局部特征提取能力和Transformer的全局依赖建模能力
- **高泛化性设计**：无需为每个测站单独建模，一个模型适用于所有测站和卫星
- **完整的工程流程**：包含数据生成、模型训练、测试评估和可视化工具

## 系统架构

### 模型架构

```
输入数据 → 特征编码层 → 时序特征提取层 → 特征融合层 → 分类层 → 输出
           ├─ 卫星嵌入     ├─ CNN分支        
           ├─ 位置编码     └─ Transformer分支  
           └─ 时间编码                         
```

### 关键技术

1. **特征工程**
   - 载噪比标准化（Z-score）
   - 空间位置编码（XYZ坐标转相对位置）
   - 时间周期性编码（sin/cos）
   - 卫星特性嵌入（32维可学习向量）

2. **损失函数设计**
   - Focal Loss：处理类别不平衡
   - 对比学习损失：增强特征判别性
   - 平滑性约束：保证时序预测的连续性

3. **数据增强**
   - 噪声注入
   - 时序扰动

## 项目结构

```
flex_power_detection/
├── config.py           # 配置文件
├── model.py            # 模型定义
├── generate_data.py    # 数据生成脚本
├── train.py            # 训练脚本
├── test.py             # 测试脚本
├── visualize.py        # 可视化工具
├── README.md           # 项目说明
├── data/              # 数据目录
│   ├── raw/           # 原始数据
│   └── processed/     # 处理后的数据
├── checkpoints/       # 模型检查点
├── logs/              # 训练日志
├── test_results/      # 测试结果
└── visualization_results/  # 可视化结果
```

## 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.0 (可选，用于GPU加速)
```

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖包列表：
```
torch>=1.10.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.60.0
tensorboardX>=2.4
```

### 使用流程

#### 1. 生成模拟数据

```bash
python generate_data.py
```

这将生成训练集、验证集和测试集，保存在`data/processed/`目录下。

生成的数据包括：
- 10000个样本（可调整）
- 32颗GPS卫星（G01-G32）
- 3个测站位置
- 包含Flex Power开启/关闭状态的标注

#### 2. 训练模型

```bash
python train.py
```

训练参数可在`config.py`中调整：
- 批量大小：64
- 学习率：1e-3
- 训练轮数：100
- 优化器：AdamW

训练过程中会：
- 每5个epoch保存检查点
- 自动进行早停（patience=15）
- 保存最佳模型到`checkpoints/best_model.pth`
- 记录TensorBoard日志到`logs/`

#### 3. 测试模型

```bash
python test.py
```

测试脚本会：
- 加载最佳模型
- 在测试集上评估性能
- 生成混淆矩阵和ROC曲线
- 输出详细的测试报告

#### 4. 可视化结果

```bash
python visualize.py
```

可视化工具提供：
- S2W载噪比时间序列图
- 真值和预测结果的对比标注
- 多卫星比较图
- 预测置信度分布图

### 可视化说明

在生成的时间序列图中：
- **橙色**：真值为Flex Power开启
- **红色**：模型预测为Flex Power开启
- **绿色**：正常状态（无Flex Power）
- **紫色**：真值和预测都为Flex Power开启

## 数据格式

### 输入数据格式

| 数据类型 | 格式 | 说明 |
|---------|------|------|
| S2W载噪比 | double | 当前历元S2W频段载噪比值 |
| S1C载噪比 | double | 当前历元S1C频段载噪比值 |
| S2W-S1C差分 | double | 载噪比差值 |
| S2W时序窗口 | array[5] | 前5个历元的S2W值 |
| S1C时序窗口 | array[5] | 前5个历元的S1C值 |
| 差分时序窗口 | array[5] | 前5个历元的差分值 |
| 测站坐标 | XYZ[3] | ECEF坐标系 |
| 卫星位置 | XYZ[3] | ECEF坐标系 |
| 本地时间 | [Y,M,D,h,m,s] | 年月日时分秒 |
| 卫星号 | G01-G32 | GPS卫星PRN号 |

### 输出格式

模型输出二分类结果：
- 0：Flex Power未开启
- 1：Flex Power已开启

## 性能指标

在模拟数据上的典型性能：
- 准确率：~92%
- 精确率：~89%
- 召回率：~85%
- F1分数：~87%
- ROC AUC：~0.95

## 配置说明

主要配置项（`config.py`）：

### 数据配置
- `window_size`: 时序窗口大小（默认5）
- `satellite_num`: GPS卫星数量（32）
- `train_ratio`: 训练集比例（0.7）

### 模型配置
- `satellite_embedding_dim`: 卫星嵌入维度（32）
- `transformer_n_heads`: 注意力头数（4）
- `transformer_n_layers`: Transformer层数（2）

### 训练配置
- `batch_size`: 批量大小（64）
- `num_epochs`: 训练轮数（100）
- `learning_rate`: 学习率（1e-3）
- `early_stopping_patience`: 早停耐心值（15）

## 扩展和改进

### 可能的改进方向

1. **数据增强**
   - 添加更多的时序扰动策略
   - 引入对抗训练

2. **模型优化**
   - 尝试更深的网络结构
   - 引入注意力机制的变体（如Cross-Attention）
   - 使用预训练模型进行迁移学习

3. **特征工程**
   - 添加更多的统计特征
   - 引入频域特征
   - 考虑天气等外部因素

4. **实时部署**
   - 模型量化和压缩
   - 边缘设备优化
   - 流式处理支持

### 真实数据适配

将模型应用于真实数据时，需要：

1. **数据预处理**
   - 解析OBS观测文件获取载噪比数据
   - 从SP3文件获取卫星位置
   - 处理数据缺失和异常值

2. **标注策略**
   - 使用已知的Flex Power开启区域进行初始标注
   - 结合多测站观测进行交叉验证
   - 人工审核边界区域

3. **模型微调**
   - 使用真实数据进行迁移学习
   - 调整类别权重以适应实际的正负样本比例
   - 根据实际噪声水平调整数据增强策略

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 使用梯度累积
   - 切换到CPU训练

2. **训练不收敛**
   - 检查数据标准化
   - 调整学习率
   - 增加数据量

3. **过拟合**
   - 增加dropout率
   - 使用更强的数据增强
   - 减少模型复杂度

## 贡献指南

欢迎贡献代码、报告问题或提出建议。请通过以下方式参与：

1. Fork本项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至: [your-email@example.com]

## 致谢

感谢所有为本项目做出贡献的开发者和研究人员。

---

*最后更新: 2025年1月*