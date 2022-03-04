from typing import *
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel,BertModel
from transformers.file_utils import ModelOutput
from torchblocks.layers.position import PositionalEncoding
from torchblocks.losses.cross_entropy import MultiLabelCategoricalCrossEntropy


@dataclass
class GlobalPointOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: List[List[List[str]]] = None
    groundtruths: List[List[List[str]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertGlobalPointerForSeqLabel(BertPreTrainedModel):  # config.pe_dim=64
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pe = PositionalEncoding(d_model=config.pe_dim, max_len=config.max_seq_length)
        self.linear = nn.Linear(config.hidden_size, config.num_labels * config.pe_dim * 2)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        batch_size, seq_length = input_ids.shape
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.linear(sequence_output)
        sequence_output = torch.split(sequence_output, self.config.pe_dim * 2, dim=-1)
        # (batch_size, seq_len, num_labels, pe_dim * 2)
        sequence_output = torch.stack(sequence_output, dim=-2)
        # query, key: (batch_size, seq_len,num_labels,  pe_dim)
        query, key = sequence_output[..., :self.config.pe_dim], sequence_output[..., self.config.pe_dim:]
        if self.config.use_rope:
            pos_emb = self.pe(batch_size, seq_length)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-query[..., 1::2], query[..., ::2]], -1)
            qw2 = qw2.reshape(query.shape)
            query = query * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-key[..., 1::2], key[..., ::2]], -1)
            kw2 = kw2.reshape(key.shape)
            key = key * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmhd,bnhd->bhmn', query, key)  # logits: (batch_size, ent_type_size, seq_len, seq_len)
        # 构建mask
        extended_attention_mask = attention_mask[:, None, None, :] * torch.triu(torch.ones_like(logits))
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e12
        logits += extended_attention_mask
        logits /= self.config.pe_dim ** 0.5
        loss, groudtruths, predictions = None, None, None
        if labels is not None:
            loss_fct = MultiLabelCategoricalCrossEntropy()
            loss = loss_fct(preds=logits.reshape(batch_size * self.config.num_labels, -1),
                            target=labels.reshape(batch_size * self.config.num_labels, -1))
            if not self.training:  # 训练时无需解码
                groudtruths = self.decode(logits=labels)
        if not self.training:  # 训练时无需解码
            predictions = self.decode(logits=logits)
        return GlobalPointOutput(loss=loss,
                                 logits=logits,
                                 predictions=predictions,
                                 groundtruths=groudtruths)

    def decode(self, logits):
        all_entity_list = []
        batch_size = logits.size(0)
        for bs in range(batch_size):
            entity_list = []
            _logits = logits[bs].float()
            _logits[:,[0,-1]] -= torch.tensor(float("inf"))
            _logits[:,:,[0,-1]] -= torch.tensor(float("inf"))
            for label_id, start_idx, end_idx in zip(*torch.where(_logits > self.config.decode_thresh)):
                label_id, start_idx, end_idx = label_id.item(), start_idx.item(), end_idx.item()
                label = self.config.id2label[label_id]
                entity_list.append([start_idx - 1, end_idx - 1, label])
            all_entity_list.append(entity_list)
        # import pdb
        # pdb.set_trace()
        return all_entity_list
