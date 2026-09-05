
# 左右半球通道对称交换（中线不动）→ 二分类：是否交换
from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# —— 32 通道（0-based）索引 —— #
# ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6','Cz','C3','C4','T3','T4','A1','A2','CP1','CP2','CP5','CP6','Pz','P3','P4','T5','T6','PO3','PO4','Oz','O1','O2']
DEFAULT_PAIRS_32: List[Tuple[int, int]] = [
    (0, 1),    # Fp1 ↔ Fp2
    (3, 4),    # F3  ↔ F4
    (5, 6),    # F7  ↔ F8
    (7, 8),    # FC1 ↔ FC2
    (9,10),    # FC5 ↔ FC6
    (12,13),   # C3  ↔ C4
    (14,15),   # T3  ↔ T4
    (16,17),   # A1  ↔ A2
    (18,19),   # CP1 ↔ CP2
    (20,21),   # CP5 ↔ CP6
    (23,24),   # P3  ↔ P4
    (25,26),   # T5  ↔ T6
    (27,28),   # PO3 ↔ PO4
    (30,31),   # O1  ↔ O2
]
DEFAULT_MIDLINE_32 = [2, 11, 22, 29]  # Fz, Cz, Pz, Oz

class JigsawPretext(nn.Module):
    """
    使用方法：
      1) 先把输入展平成 (B, C, T)，调用 make_symmetry(x_flat)。其中 C 应与上面的 32 通道一致。
      2) 得到 (x_out, labels)：labels=0 表示未交换，1 表示已交换。
      3) 将 x_out 送入你的 backbone → 得到特征 feats_flat:(B, D)；
         再用 loss_from_features(feats_flat, labels) 得到交叉熵损失。
    """
    def __init__(self,
                 pairs: Optional[List[Tuple[int,int]]] = None,
                 midline: Optional[List[int]] = None,
                 hidden_dim: int = 512):
        super().__init__()
        self.pairs = pairs if pairs is not None else DEFAULT_PAIRS_32
        self.midline = set(midline if midline is not None else DEFAULT_MIDLINE_32)
        self.hidden_dim = hidden_dim
        self.classifier: Optional[nn.Sequential] = None
        self._in_dim: Optional[int] = None

    @torch.no_grad()
    def make_symmetry(self, x_flat: torch.Tensor):
        """
        输入  x_flat: (B, C, T)   （C 按本文件 32 通道顺序排列，0-based）
        输出  x_out:  (B, C, T)； labels: (B,)  in {0:未交换, 1:已交换}
        逻辑  以 50% 概率对样本进行左右半球成对通道互换；中线通道保持不变。
        """
        if x_flat.dim() != 3:
            raise ValueError(f"x_flat must be (B,C,T), got {tuple(x_flat.shape)}")
        B, C, T = x_flat.shape
        out = x_flat.clone()
        labels = torch.zeros(B, dtype=torch.long, device=x_flat.device)

        # 仅使用索引在范围内的成对通道
        pairs = [(i, j) for (i, j) in self.pairs if (i < C and j < C)]
        for b in range(B):
            do_swap = torch.randint(0, 2, (1,), device=x_flat.device).item()  # 0/1
            if do_swap == 1:
                for i, j in pairs:
                    if (i in self.midline) or (j in self.midline):
                        continue
                    tmp = out[b, i].clone()
                    out[b, i] = out[b, j]
                    out[b, j] = tmp
                labels[b] = 1
        return out, labels

    def _ensure_head(self, in_dim: int):
        if (self.classifier is None) or (self._in_dim != in_dim):
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 2),  # 0/1 是否交换
            )
            self._in_dim = in_dim

    def loss_from_features(self, feats_flat: torch.Tensor, swap_labels: torch.Tensor):
        """
        feats_flat: (B, D) 由 backbone 输出展平
        swap_labels: (B,) in {0,1}
        返回 (loss, preds)
        """
        self._ensure_head(feats_flat.size(1))
        logits = self.classifier(feats_flat)
        loss = F.cross_entropy(logits, swap_labels)
        preds = logits.argmax(dim=-1)
        return loss, preds
