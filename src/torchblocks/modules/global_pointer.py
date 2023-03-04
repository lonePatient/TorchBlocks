import copy
import torch
import torch.nn as nn
from .position import SinusoidalPositionEmbedding


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(self, head_size, inner_dim, hidden_size, rope=True):
        super(GlobalPointer, self).__init__()
        self.rope = rope
        self.inner_dim = inner_dim
        self.head_size = head_size  # num_label
        self.dense = nn.Linear(hidden_size, self.head_size * self.inner_dim * 2)
        self.pos_emd = SinusoidalPositionEmbedding(self.inner_dim, 'zero')

    def multilabel_categorical_crossentropy(self, targets, entity_score):
        """Multi-label cross entropy loss.
        """
        entity_score = (1 - 2 * targets) * entity_score  # -1 -> pos classes, 1 -> neg classes
        entity_score_neg = entity_score - targets * 1e12  # mask the pred outputs of pos classes
        entity_score_pos = (
                entity_score - (1 - targets) * 1e12
        )  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(entity_score[..., :1])
        entity_score_neg = torch.cat([entity_score_neg, zeros], dim=-1)
        entity_score_pos = torch.cat([entity_score_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(entity_score_neg, dim=-1)
        pos_loss = torch.logsumexp(entity_score_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def compute_loss(self, entity_score, targets) -> torch.Tensor:
        """
        targets : (batch_size, num_classes, seq_len, seq_len)
        entity_score : (batch_size, num_classes, seq_len, seq_len)
        """
        batch_size, num_classes = entity_score.shape[:2]
        targets = targets.reshape(batch_size * num_classes, -1)
        entity_score = entity_score.reshape(batch_size * num_classes, -1)
        loss = self.multilabel_categorical_crossentropy(targets, entity_score)
        return loss

    def add_position_embedding(self, input_embed, cos_pos, sin_pos):
        tran_embed = torch.stack([-input_embed[..., 1::2], input_embed[..., ::2]], 4)
        tran_embed = torch.reshape(tran_embed, input_embed.shape)
        output_embed = input_embed * cos_pos + tran_embed * sin_pos
        return output_embed

    def forward(self, sequence_output, mask=None):
        batch_size = sequence_output.size()[0]
        seq_len = sequence_output.size()[1]
        outputs = self.dense(sequence_output)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        outputs = torch.stack(outputs, dim=-2)
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        # 分出qw和kw
        # RoPE编码
        if self.rope:
            pos_emb = self.pos_emd(outputs)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw = self.add_position_embedding(qw, cos_pos, sin_pos)
            kw = self.add_position_embedding(kw, cos_pos, sin_pos)
        # 计算内积
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # padding mask
        pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.head_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        return logits / self.inner_dim ** 0.5
