import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel


class BertWithMDP(BertPreTrainedModel):
    '''
    对每一层的[CLS]向量进行weight求和，以及添加multi-sample dropout
    '''

    def __init__(self, config):
        config.output_hidden_states = True
        super(BertWithMDP, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.high_dropout = nn.Dropout(0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_layers = outputs[2]
        cls_outputs = torch.stack([self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2)
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)
        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([self.classifier(self.high_dropout(cls_output)) for _ in range(5)], dim=0),
                            dim=0)
        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
