import torch
import torch.nn as nn
from typing import *
from torch.nn import CrossEntropyLoss
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from transformers import BertPreTrainedModel, BertModel

@dataclass
class SequenceLabelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: List[List[List[str]]] = None
    groundtruths: List[List[List[str]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertSoftmaxForSeqLabel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None
                ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss, groundtruths, predictions = None, None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            if not self.training:
                groundtruths = self.decode(labels, attention_mask, is_logits=False)
        if not self.training:  # 训练时无需解码
            predictions = self.decode(logits, attention_mask, is_logits=True)
        return SequenceLabelingOutput(
            loss=loss,
            logits=logits,
            predictions=predictions,
            groundtruths=groundtruths,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, logits, mask, is_logits=False) -> List[List[List[str]]]:
        decode_ids = logits
        if is_logits:
            decode_ids = torch.argmax(logits, -1)  # (batch_size, seq_length)
        decode_labels = []
        for ids, mask in zip(decode_ids, mask):
            decode_label = [self.config.id2label[id.item()] for id, m in zip(ids, mask) if m > 0][1:-1]  # [CLS], [SEP]
            decode_labels.append(decode_label)
        return decode_labels
