import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


class BertForSiameseModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSiameseModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seq_relationship = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.init_weights()

    def forward(self, a_input_ids, b_input_ids,
                a_token_type_ids=None,
                b_token_type_ids=None,
                a_attention_mask=None,
                b_attention_mask=None,
                labels=None):
        a_outputs = self.bert(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask)
        b_outputs = self.bert(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)
        # concat
        concated_pooled_output = torch.cat(
            [a_outputs[1], b_outputs[1], torch.abs(a_outputs[1] - b_outputs[1])], dim=1)

        concated_pooled_output = self.dropout(concated_pooled_output)
        logits = self.seq_relationship(concated_pooled_output)
        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
