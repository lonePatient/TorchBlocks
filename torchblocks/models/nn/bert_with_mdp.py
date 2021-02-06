import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from torchblocks.models.layers.dropouts import MultiSampleDropout


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

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.classifier = MultiSampleDropout(config.hidden_size, config.num_labels, K=5, p=0.5)
        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_layers = outputs[2]
        cls_outputs = torch.stack([self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2)
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)
        logits = self.classifier(cls_output)
        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
