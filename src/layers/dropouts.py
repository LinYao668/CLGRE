import torch
import torch.nn as nn


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.5):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, inputs):  # sourcery skip: avoid-builtin-shadow
        inputs = inputs.unsqueeze(2)  # (N, T, 1, K)
        inputs = inputs.permute(0, 3, 2, 1)  # (N, K, 1, T)
        inputs = super(SpatialDropout, self).forward(inputs)  # (N, K, 1, T), some features are masked
        inputs = inputs.permute(0, 3, 2, 1)  # (N, T, 1, K)
        return inputs.squeeze(2)  # (N, T, K)


class MultiSampleDropout(nn.Module):
    """
    # multisample dropout (wut): https://arxiv.org/abs/1905.09788
    """

    def __init__(self, hidden_size, num_labels, K=5, p=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs):
        logits = torch.stack([self.classifier(self.dropout(inputs)) for _ in range(self.K)], dim=0)
        logits = torch.mean(logits, dim=0)
        return logits


class TimestepDropout(torch.nn.Dropout):
    r"""
    传入参数的shape为 ``(batch_size, num_timesteps, embedding_dim)``
    使用同一个shape为 ``(batch_size, embedding_dim)`` 的mask在每个timestamp上做dropout。
    """

    def forward(self, x):
        dropout_mask = x.new_ones(x.shape[0], x.shape[-1])
        torch.nn.functional.dropout(dropout_mask, self.p, self.training, inplace=True)
        dropout_mask = dropout_mask.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        if not self.inplace:
            return x * dropout_mask
        x *= dropout_mask
        return x
