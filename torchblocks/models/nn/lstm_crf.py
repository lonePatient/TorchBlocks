import torch.nn as nn
from torch.nn import LayerNorm
from torchblocks.models.layers.crf import CRF
from torchblocks.models.layers.dropouts import SpatialDropout
from torchblocks.models.bases.model_base import TrainModel


class LSTMCRF(TrainModel):
    def __init__(self, config):
        super(LSTMCRF, self).__init__(config)
        self.emebdding_size = config.embedding_size
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.bilstm = nn.LSTM(input_size=config.embedding_size,
                              hidden_size=config.hidden_size,
                              num_layers=config.num_hidden_layers,
                              dropout=config.lstm_dropout_prob,
                              batch_first=True,
                              bidirectional=True)
        self.dropout = SpatialDropout(config.hidden_dropout_prob)
        self.layer_norm = LayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids,
                attention_mask=None,
                labels=None):
        embs = self.embedding(input_ids)
        embs = self.dropout(embs)
        embs = embs * attention_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output = self.layer_norm(seqence_output)
        logits = self.classifier(seqence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs
