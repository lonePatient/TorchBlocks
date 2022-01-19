import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class Pooling(nn.Module):
    def __init__(self, hidden_size, pooling_mode='cls', last_layers=None):
        super(Pooling, self).__init__()
        assert pooling_mode in ['mean', 'max', 'cls', 'mean_sqrt']
        self.hidden_size = hidden_size
        self.last_layers = last_layers
        self.pooling_mode = pooling_mode
        self.pooling_output_dimension = hidden_size if last_layers is None else hidden_size * last_layers

    def forward(self, features, attention_mask):
        sequence_outputs = features['last_hidden_state']
        cls_outputs = features['pooler_output']
        hidden_outputs = features['hidden_states']
        if self.last_layers is not None:
            sequence_outputs = torch.cat([hidden_outputs[-i] for i in range(1, self.last_layers + 1)], dim=-1)
        if self.pooling_mode == 'cls':
            vectors = cls_outputs
        if self.pooling_mode == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_outputs.size()).float()
            sequence_outputs[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            vectors = torch.max(sequence_outputs, 1)[0]
        if self.pooling_mode in ['mean', 'mean_sqrt']:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_outputs.size()).float()
            sum_embeddings = torch.sum(sequence_outputs * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            if self.pooling_mode == 'mean':
                vectors = sum_embeddings / sum_mask
            if self.pooling_mode == 'mean_sqrt':
                vectors = sum_embeddings / torch.sqrt(sum_mask)
        return vectors

    def get_pooling_output_dimension(self):
        return self.pooling_output_dimension


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.pooling = Pooling(hidden_size=config.hidden_size,
                               pooling_mode=config.pooling_mode,
                               last_layers=config.last_layers)
        pooling_output_dimension = self.pooling.get_pooling_output_dimension()
        self.classifier = nn.Linear(pooling_output_dimension, config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None,
                ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self.pooling(outputs, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
