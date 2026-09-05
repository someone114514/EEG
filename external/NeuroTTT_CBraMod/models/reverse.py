# # CBraMod/models/reverse.py
# import random
# import torch
# import torch.nn as nn

# class ReversePretext(nn.Module):
#     """
#     反转时序的自监督任务：
#     - 随机决定是否对每个样本进行“部分通道”的时间维反转（last dim）。
#     - 若反转：随机选择 >50% 的通道进行反转；否则不反转。
#     - 让模型预测是否发生了反转：0=未反转，1=已反转（部分或全部）。
#     - 分类器输入特征维度与其它 pretext 一致（默认使用通道平均后的 3*200）。
#     """
#     def __init__(self, input_dim: int):
#         super().__init__()
#         self.classifier = nn.Linear(in_features=input_dim, out_features=2)

#     def flip_all_or_not(self, x: torch.Tensor):
#         """
#         参数:
#             x: Tensor，形状 (B, C, T)，注意这里的 T = seg * pts
#         returns:
#             x_out: 可能被部分通道反转的 x
#             label: 若 B==1，则返回 int；否则返回长度为 B 的 python list[int]
#         """
#         assert x.dim() == 3, f"Expected x shape (B, C, T), got {tuple(x.shape)}"
#         B, C, T = x.shape
#         x_out = x.clone()
#         labels = []

#         # 计算严格大于 50% 的通道下界
#         min_k = C // 2 + 1  # e.g., C=64 -> 33, C=63 -> 32

#         for i in range(B):
#             if random.random() < 0.5:
#                 # 在 [min_k, C] 之间随机选择一个通道数 k
#                 k = random.randint(min_k, C)
#                 # 从所有通道里无放回随机选取 k 个通道索引（保持与 x 同设备）
#                 idx = torch.randperm(C, device=x_out.device)[:k]
#                 # 仅对这些通道在时间维做反转
#                 x_out[i, idx] = torch.flip(x_out[i, idx], dims=[-1])  # 参考: torch.flip 文档
#                 labels.append(1)
#             else:
#                 labels.append(0)

#         return x_out, (labels[0] if B == 1 else labels)

# CBraMod/models/reverse.py
import random
import torch
import torch.nn as nn

class ReversePretext(nn.Module):
    """
    反转时序的自监督任务：
    - 随机决定是否对每个样本的“所有通道”一起做时间维度反转（last dim）。
    - 让模型预测是否发生了反转：0=未反转，1=已反转。
    - 分类器输入特征维度与其它 pretext 一致（默认使用通道平均后的 3*200）。
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(in_features=input_dim, out_features=2)

    def flip_all_or_not(self, x: torch.Tensor):
        """
        参数:
            x: Tensor，形状 (B, C, T)，注意这里的 T = seg * pts
        returns:
            x_out: reversed x
            label: 若 B==1，则返回 int；否则返回长度为 B 的 python list[int]
        """
        assert x.dim() == 3, f"Expected x shape (B, C, T), got {tuple(x.shape)}"
        B, C, T = x.shape
        x_out = x.clone()
        labels = []
        for i in range(B):
            if random.random() < 0.5:
                # 反转时间维
                x_out[i] = torch.flip(x_out[i], dims=[-1])
                labels.append(1)
            else:
                labels.append(0)
        return x_out, (labels[0] if B == 1 else labels)

