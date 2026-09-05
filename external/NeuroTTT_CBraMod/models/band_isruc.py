import torch
import torchaudio
import random
from typing import List, Tuple

class BandPretext(torch.nn.Module):
    """
    Band-reject pretext head:
    - 自动根据 sfreq 和（可选）max_freq 过滤无效频带
    - 每个样本随机选择一个频带并做带阻
    - 返回 filtered_x, band_labels
    - 提供一个简单的线性分类器用于做带别预测
    """
    def __init__(self, input_dim: int,
                 bands: List[Tuple[float, float]] = None,
                 min_freq: float = 0.3,
                 max_freq: float = 35.0,
                 edge_margin_hz: float = 0.5,
                 min_q: float = 0.5):
        super().__init__()
        # δ, θ, α, β1, β2（把 β 拆两段让上沿不超 35）
        if bands is None:
            bands = [(0.3, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 35.0)]
        self._orig_bands = bands
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.edge_margin_hz = edge_margin_hz  # 距奈奎斯特与 0 Hz 留一点安全边
        self.min_q = min_q

        self.classifier = torch.nn.Linear(
            in_features=input_dim,
            out_features=len(bands),
        )

    def _valid_bands(self, sfreq: float) -> List[Tuple[float, float]]:
        nyq = sfreq / 2.0
        upper_limit = min(self.max_freq, nyq - self.edge_margin_hz)
        lower_limit = max(self.min_freq, 0.0 + self.edge_margin_hz)
        bands = []
        for lo, hi in self._orig_bands:
            lo2 = max(lo, lower_limit)
            hi2 = min(hi, upper_limit)
            if hi2 - lo2 > 0.0:
                bands.append((lo2, hi2))
        # 至少保留一个频带
        if not bands:
            # 退化为把整个可分析频段当一个带
            bands = [(lower_limit, upper_limit)]
        return bands

    @torch.no_grad()
    def reject_band(self, x: torch.Tensor, sfreq: torch.Tensor):
        """
        x: (B, C, T) 或 (B, T)；带阻作用在时间维 T 上，C 视为批的额外维（广播）。
        sfreq: 标量或形如 [1] 的张量（Hz）
        返回:
            filtered_x: 与 x 同形状
            band_labels: (B,) 每个样本被挖掉的频带索引
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,T)
            squeeze_back = True
        else:
            squeeze_back = False

        B, C, T = x.shape
        sr = float(sfreq[0].item()) if sfreq.numel() > 0 else float(sfreq)

        bands = self._valid_bands(sr)
        num_bands = len(bands)

        # 结果缓存
        y = torch.empty_like(x)
        labels = torch.empty(B, dtype=torch.long, device=x.device)

        for b in range(B):
            idx = random.randrange(num_bands)
            lo, hi = bands[idx]
            cf = (lo + hi) / 2.0
            bw = max(hi - lo, 1e-6)
            Q  = max(cf / bw, self.min_q)

            # torchaudio 的 biquad 支持对前导维广播；我们对 (C,T) 做同一个滤波器
            # 输入 (..., time)；把 (C,T) 看作批内多通道
            xb = x[b]                      # (C,T)
            yb = torchaudio.functional.bandreject_biquad(
                    xb, sample_rate=sr, central_freq=cf, Q=Q
                 )
            y[b] = yb
            labels[b] = idx

        if squeeze_back:
            y = y.squeeze(1)  # (B,T)
        return y, labels

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        feats: (B, D) —— 你从 backbone 抽出来的表征
        labels: (B,) —— reject_band 返回的带别
        """
        logits = self.classifier(feats)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels
