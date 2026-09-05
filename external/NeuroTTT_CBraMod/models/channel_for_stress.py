# CBraMod/models/channel.py
import torch
import random

# 你的通道顺序（下标从 0 开始）
DEFAULT_SELECTED_CHANNELS = [
    'EEG Fp1','EEG Fp2','EEG F3','EEG F4','EEG F7','EEG F8',
    'EEG T3','EEG T4','EEG C3','EEG C4','EEG T5','EEG T6',
    'EEG P3','EEG P4','EEG O1','EEG O2','EEG Fz','EEG Cz',
    'EEG Pz','EEG A2-A1'
]

AP_PAIRS_BY_NAME = [
    ("EEG Fp1", "EEG O1"),   # 前额极 ↔ 枕
    ("EEG Fp2", "EEG O2"),
    ("EEG F3",  "EEG P3"),   # 额 ↔ 顶（左）
    ("EEG F4",  "EEG P4"),   # 额 ↔ 顶（右）
    ("EEG T3",  "EEG T5"),   # 前颞 ↔ 后颞（左）
    ("EEG T4",  "EEG T6"),   # 前颞 ↔ 后颞（右）
    ("EEG Fz",  "EEG Pz"),   # 中线 额 ↔ 中线 顶
    # 其余：C3/C4/Cz 无直接“后”对位；A2-A1 为参考，不参与交换
]

class ChannelPretext(torch.nn.Module):
    """
    随机选择一个前↔后通道对并交换两通道的信号；分类头预测被交换的是哪一对。
    x 形状: (B, C, T)
    """
    def __init__(self, input_dim, channel_names=None, selected_channels=None):
        super().__init__()
        if channel_names is None:
            channel_names = selected_channels or DEFAULT_SELECTED_CHANNELS

        # 名称→索引
        name2idx = {ch: i for i, ch in enumerate(channel_names)}

        # 只保留两端都存在于当前阵列的成对
        self.ap_pairs = [(name2idx[a], name2idx[b])
                         for (a, b) in AP_PAIRS_BY_NAME
                         if a in name2idx and b in name2idx]

        assert len(self.ap_pairs) >= 1, "未找到可用的前↔后通道对，请检查通道名。"
        self.num_pairs = len(self.ap_pairs)

        # 分类头：类别数=成对数量
        self.classifier = torch.nn.Linear(in_features=input_dim, out_features=self.num_pairs)

    @torch.no_grad()
    def swap_one_ap_pair(self, x):
        """
        随机抽取一对 (front, back) 并交换它们的波形。
        输入 x: (B, C, T)
        返回: x_swapped, pair_label (0..num_pairs-1)
        """
        B, C, T = x.shape
        pair_label = random.randrange(self.num_pairs)
        i_front, i_back = self.ap_pairs[pair_label]

        x_swapped = x.clone()
        # 交换这两个通道（整段时间）
        tmp = x_swapped[:, i_front, :].clone()
        x_swapped[:, i_front, :] = x_swapped[:, i_back, :]
        x_swapped[:, i_back, :] = tmp
        return x_swapped, pair_label

        # 若想“一次交换所有前↔后成对通道”，用下面两行替换上面的交换逻辑：
        # x_swapped = x.clone()
        # for i_front, i_back in self.ap_pairs:
        #     x_swapped[:, [i_front, i_back], :] = x_swapped[:, [i_back, i_front], :]
        # return x_swapped, -1  # 这时可改成二分类：是否进行了全体交换

    def forward(self, swapped_x, pair_label):
        """
        分类预测被交换的是哪一对。
        swapped_x: 经过 swap_one_ap_pair 处理后的张量
        pair_label: 对应的成对索引（0..num_pairs-1）
        """
        logits = self.classifier(swapped_x)  # (B, num_pairs)
        labels = torch.full((swapped_x.shape[0],), pair_label,
                            dtype=torch.long, device=swapped_x.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels
