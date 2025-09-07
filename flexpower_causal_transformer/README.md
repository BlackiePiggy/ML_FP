# 端到端 Flex Power 检测 — Causal Transformer（逐点级）

这是一个 **可直接用于生产的最小脚手架**，用于判断当前历元 `t` 是否处于 **Flex Power 开启** 状态，  
采用 **Causal Transformer** 并在台站 / 接收机 / 天线 / 星座 / 卫星号等元信息上进行 **FiLM** 域调制。  
整个流程是 **端到端** 的，**无需手工特征工程**。

---

## 关键思想
- 输入：短时间因果窗口 `[t-K … t]` 上的原始序列（例如：4 分钟 @ 1 Hz ⇒ 240 步）：  
  - 连续特征（默认 6 个）：`S1C_CN0, S2W_CN0, elevation, azimuth, dElev_dt, mask`。  
  - 离散元信息：`station_id, receiver_id, antenna_id, constellation_id, prn_id`。  
- 嵌入层：  
  - 连续通道通过线性投影到 `d_model`。  
  - 离散 ID 使用可学习 embedding，并通过 **FiLM** 融合。  
- Causal Transformer：  
  - N 层因果自注意力 + 残差连接。  
  - 可学习的位置编码。  
- 输出层：读取 **最后一个时间步**，得到逐点级 logit → `sigmoid`。  
- 损失函数：默认 Focal Loss，可选 BCE。  
- 评价指标：Accuracy / Precision / Recall / F1。（如需可加入 AUROC / AUPRC）

---

## 快速开始（合成数据）
```bash
# （可选）创建虚拟环境；如需安装 torch
# pip install torch pyyaml

python train.py            # 在合成数据集上训练
python export_onnx.py      # 导出 ONNX 模型用于部署
