# temporal.py —— Temporal Jigsaw (4 chunks) pretext head
import torch
import torch.nn as nn

class TemporalPretext(nn.Module):
    """
    Binary classification: predict whether a temporal rearrangement happened.
    Label convention:
        0 -> original order
        1 -> rearranged (shuffled, guaranteed != original)
    """
    def __init__(self, input_dim: int, num_chunks: int = 2, p_shuffle: float = 0.5):
        super().__init__()
        assert num_chunks >= 2, "num_chunks must be >= 2"
        self.input_dim = input_dim
        self.num_chunks = int(num_chunks)
        self.p_shuffle = float(p_shuffle)

        # 与其它 pretext 一致，用 Linear + CE，两类
        self.classifier = nn.Linear(in_features=input_dim, out_features=2)

    @staticmethod
    def _split_chunks(x: torch.Tensor, n: int):
        """
        x: [B, C, T] -> list of n contiguous chunks [B, C, Ti].
        The last chunk absorbs the remainder.
        """
        B, C, T = x.shape
        base = T // n
        sizes = [base] * n
        sizes[-1] += T - base * n
        idx = [0]
        for s in sizes:
            idx.append(idx[-1] + s)
        return [x[:, :, idx[i]:idx[i+1]] for i in range(n)]

    @staticmethod
    def _non_identity_perm(n: int):
        """Sample a permutation definitely different from identity."""
        while True:
            p = torch.randperm(n).tolist()
            if any(p[i] != i for i in range(n)):
                return p

    def rearrange_or_not(self, x: torch.Tensor):
        """
        x: [B, C, T]
        Randomly decide to shuffle or not; when shuffling, permute 4 chunks
        with a permutation != identity.
        Returns:
            xr: [B, C, T]  (original or rearranged)
            label: int (0 original, 1 rearranged)
        """
        assert x.dim() == 3, "expected [B, C, T]"
        B, C, T = x.shape
        do_shuffle = (torch.rand(1).item() < self.p_shuffle)

        if not do_shuffle:
            return x, 0  # original

        chunks = self._split_chunks(x, self.num_chunks)   # length=4
        perm = self._non_identity_perm(self.num_chunks)   # != [0,1,2,3]
        xr = torch.cat([chunks[i] for i in perm], dim=2)
        return xr, 1  # rearranged

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        feats: [B, D] (与你项目其它 pretext 一样，backbone+池化后的特征)
        labels: [B] long tensor with values in {0,1}
        """
        logits = self.classifier(feats)              # [B, 2]
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)  # [B]
        return loss, preds, labels
