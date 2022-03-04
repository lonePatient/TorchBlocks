import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from dataclasses import dataclass
from collections import defaultdict
from torchblocks.losses.span_loss import SpanLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.file_utils import ModelOutput
from torchblocks.layers.linears import PoolerStartLogits, PoolerEndLogits


@dataclass
class SpanOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    predictions: List[List[Tuple[int, int, int]]] = None
    groundtruths: List[List[Tuple[int, int, int]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertSpanForSeqLabel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, config.num_labels)
        self.end_fc = PoolerEndLogits(config.hidden_size + config.num_labels, config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                start_positions=None,
                end_positions=None,
                ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if self.training:
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            label_logits = torch.zeros([batch_size, seq_len, self.config.num_labels])
            label_logits = label_logits.to(input_ids.device)
            label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
        else:
            label_logits = F.softmax(start_logits, -1)
        end_logits = self.end_fc(sequence_output, label_logits)
        loss, predictions, groundtruths = None, None, None
        if start_positions is not None and end_positions is not None:
            loss_fct = SpanLoss()
            loss = loss_fct(preds=(start_logits, end_logits),
                            target=(start_positions, end_positions),
                            masks=attention_mask)
            if not self.training:  # 训练时无需解码
                groundtruths = self.decode(
                    start_positions, end_positions, attention_mask, is_logits=False
                )
        if not self.training:  # 训练时无需解码
            predictions = self.decode(
                start_logits, end_logits, attention_mask,
                start_thresh=getattr(self.config, "start_thresh", 0.0),
                end_thresh=getattr(self.config, "end_thresh", 0.0),
            )
        return SpanOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            predictions=predictions,
            groundtruths=groundtruths,
            hidden_states=None,
            attentions=None
        )

    def decode(self, start_logits_or_labels, end_logits_or_labels, sequence_mask,
               start_thresh=0.0, end_thresh=0.0, is_logits=True, **kwargs):
        """
        Params:
            start_logits_or_labels: tensor(batch_size, sequence_length, num_labels)
            end_logits_or_labels: tensor(batch_size, sequence_length, num_labels)
            sequence_mask: tensor(batch_size, sequence_length)
        Returns:
            predictions: List[List[Tuple[int, int, int]]]
        """
        other_id = self.config.label2id["O"]
        id2label = self.config.id2label
        max_span_length = kwargs.get("max_span_length", float("inf"))
        if is_logits:  # 复用decode
            # TODO: 概率先判断是否为实体
            start_probs = start_logits_or_labels.softmax(dim=-1)  # (batch_size, sequence_length, num_labels)
            other_probs = start_probs[..., other_id]  # (batch_size, sequence_length)
            other_probs = torch.where(other_probs < start_thresh,
                                      torch.zeros_like(other_probs), other_probs)
            start_probs[..., other_id] = other_probs
            start_probs, start_labels = start_probs.max(dim=-1)

            end_probs = end_logits_or_labels.softmax(dim=-1)  # (batch_size, sequence_length, num_labels)
            other_probs = end_probs[..., other_id]  # (batch_size, sequence_length)
            other_probs = torch.where(other_probs < end_thresh,
                                      torch.zeros_like(other_probs), other_probs)
            end_probs[..., other_id] = other_probs
            end_probs, end_labels = end_probs.max(dim=-1)

        else:
            start_labels, end_labels = start_logits_or_labels, end_logits_or_labels
        decode_labels = []
        batch_size = sequence_mask.size(0)
        for i in range(batch_size):
            decode_labels.append([])
            label_start_map = defaultdict(list)  # 每种类别设置起始标志，处理实体重叠情况，如：
            # start: [0, 0, 1, 0, 2, 0, 0, 0]
            # end:   [0, 0, 0, 0, 0, 2, 1, 0]
            for pos, (s, e, m) in enumerate(zip(start_labels[i], end_labels[i], sequence_mask[i])):
                s, e, m = s.item(), e.item(), m.item()
                if m == 0: break
                if s != other_id:
                    label_start_map[s].append(pos)  # 以下两个功能：
                    # 1. 进入s类型span，以label_start_map[s]标记;
                    # 2. 若在s类型span内，但重新遇到s类型span起始时，追加位置
                if e != other_id:  # 在e类型span内（已包括处理单个token的实体）
                    for start in label_start_map[e]:
                        start, end, label = start - 1, pos, id2label[e]  # [CLS]
                        if end - start < max_span_length:
                            decode_labels[-1].append((start, end, label))  # 遇到结束位置，新建span
                    label_start_map[e] = list()
            # TODO: 强制匹配策略，即start必须匹配end
            for k, v in label_start_map.items():
                if v is not None:
                    pass
        return decode_labels
