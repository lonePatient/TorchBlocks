import torch
from typing import *
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.file_utils import ModelOutput


@dataclass
class SiameseClassificatioOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class BertForSiameseClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSiameseClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seq_relationship = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.init_weights()

    def forward(self, input_ids_a, input_ids_b,
                token_type_ids_a=None,
                token_type_ids_b=None,
                attention_mask_a=None,
                attention_mask_b=None,
                labels=None):
        outputs_a = self.bert(input_ids_a,
                              token_type_ids=token_type_ids_a,
                              attention_mask=attention_mask_a)
        outputs_b = self.bert(input_ids_b,
                              token_type_ids=token_type_ids_b,
                              attention_mask=attention_mask_b)
        pooled_output = torch.cat([outputs_a[1], outputs_b[1], torch.abs(outputs_a[1] - outputs_b[1])], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.seq_relationship(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SiameseClassificatioOutput(
            loss=loss,
            logits=logits,
        )
