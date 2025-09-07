import torch
from torch.utils.data import Dataset
import math
import random

class DummyFlexDataset(Dataset):
    """
    Synthetic dataset for demo:
    - 连续特征按时间展开，训练时会被 DataLoader 叠成 [B, L, C]
    - 类别元信息包含站点/接收机/天线/星座/PRN 的 ID
    - 标签是点级（当前时刻 t，即片段的最后一个时间步）
    """
    def __init__(self, size, seq_len=240, num_cont=6,
                 num_stations=64, num_receivers=64, num_antennas=64,
                 num_constellations=4, num_prns=64, seed=42):
        super().__init__()
        self.size = size # 数据集中样本条数（被 __len__ 返回）
        self.seq_len = seq_len # 单条样本的因果上下文长度 L（例如 240 表示 4 分钟 @ 1 Hz）
        self.num_cont = num_cont # 连续通道个数 C（这里是 6 个）
        self.num_stations = num_stations # 测站信息的词表大小
        self.num_receivers = num_receivers # 接收机信息的词表大小
        self.num_antennas = num_antennas  # 天线信息的词表大小
        self.num_constellations = num_constellations # 星座信息的词表大小
        self.num_prns = num_prns # 卫星PRN信息的词表大小
        random.seed(seed)

    def __len__(self):
        return self.size # 让 len(dataset) 返回样本总数，供 DataLoader 采样

    def __getitem__(self, idx):
        L, C = self.seq_len, self.num_cont # DataLoader 会调用它按索引取一条样本；先取出当前配置的序列长度 L 和通道数 C
        # continuous channels: [S1C_CN0, S2W_CN0, elevation, azimuth, dElev_dt, mask]
        x = torch.randn(L, C).float() # 生成一个连续特征序列张量 x，形状 [L, C]，默认正态分布. x[:,0] → S1C_CN0; x[:,1] → S2W_CN0; ...
        # impose some structure: elevation ~ [0,1], azimuth ~ [-pi,pi], mask in {0,1}
        elev = torch.linspace(0, 1, L).unsqueeze(-1)
        azim = torch.sin(torch.linspace(-math.pi, math.pi, L)).unsqueeze(-1)
        mask = (torch.rand(L, 1) > 0.05).float()  # 5% missing

        x[:, 2:3] = elev
        x[:, 3:4] = azim
        x[:, 5:6] = mask

        # synthetic label: if S2W - S1C averaged over last 30 steps > threshold, label=1
        s1c = x[:, 0]
        s2w = x[:, 1]
        diff_last = (s2w[-30:] - s1c[-30:]).mean()
        y = (diff_last > 0.2).float().unsqueeze(0)  # [1]

        # categorical meta
        station_id = torch.randint(0, self.num_stations, (1,)).long()
        receiver_id = torch.randint(0, self.num_receivers, (1,)).long()
        antenna_id = torch.randint(0, self.num_antennas, (1,)).long()
        constellation_id = torch.randint(0, self.num_constellations, (1,)).long()
        prn_id = torch.randint(0, self.num_prns, (1,)).long()

        meta = {
            "station_id": station_id,
            "receiver_id": receiver_id,
            "antenna_id": antenna_id,
            "constellation_id": constellation_id,
            "prn_id": prn_id,
        }
        return x, meta, y
