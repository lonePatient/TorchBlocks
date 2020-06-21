import torch
from torch import nn
import torch.nn.functional as F
from ..layers.dropouts import SpatialDropout
from ..layers.capsule import Capsule
from ..layers.attentions import Attention
from ..bases import TrainModel
from torch.nn import CrossEntropyLoss


class LstmGruNet(TrainModel):
    def __init__(self, config):
        super(LstmGruNet, self).__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.embedding_dropout = SpatialDropout(config.embedding_dropout)
        self.lstm = nn.LSTM(config.embedding_size, config.lstm_units, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(config.lstm_units * 2, config.gru_units, bidirectional=True, batch_first=True)
        dense_hidden_units = config.gru_units * 4
        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.classifier = nn.Linear(dense_hidden_units, config.num_labels)

    def forward(self, input_ids, labels):
        h_embedding = self.embedding(input_ids)
        h_embedding = self.embedding_dropout(h_embedding)
        h1, _ = self.lstm(h_embedding)
        h2, _ = self.gru(h1)
        # global average pooling
        avg_pool = torch.mean(h2, 1)
        # global max pooling
        max_pool, _ = torch.max(h2, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        logits = self.classifier(hidden)
        outputs = (logits,)
        if labels is not None:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
        return outputs


class LstmGruModel(TrainModel):
    def __init__(self, config):
        super(LstmGruModel, self).__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.spatial_dropout = SpatialDropout(config.embedding_dropout)
        self.lstm = nn.LSTM(config.embedding_size, config.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(config.lstm_hidden_size * 2, config.gru_hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(config.gru_hidden_size * 6, config.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, labels):
        h_embedding = self.embedding(input_ids)
        h_embedding = self.spatial_dropout(h_embedding)
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, hh_gru = self.gru(h_lstm)
        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        logits = self.classifier(conc)
        outputs = (logits,)
        if labels is not None:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
        return outputs


class LstmCapsuleAttenModel(TrainModel):

    def __init__(self, config):
        super(LstmCapsuleAttenModel, self).__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.spatial_dropout = SpatialDropout(config.embedding_dropout)
        self.lstm = nn.LSTM(config.embedding_size, config.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(config.lstm_hidden_size * 2, config.gru_hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(config.lstm_hidden_size * 2, maxlen=config.maxlen)
        self.gru_attention = Attention(config.gru_hidden_size * 2, maxlen=config.maxlen)

        self.capsule = Capsule(input_dim_capsule=config.gru_hidden_size * 2,
                               num_capsule=config.num_capsule,
                               dim_capsule=config.dim_capsule)
        self.dropout_caps = nn.Dropout(config.caps_dropout)
        self.lin_caps = nn.Linear(config.num_capsule * config.dim_capsule, config.caps_out)

        self.norm = nn.LayerNorm(config.lstm_hidden_size * 2 + config.gru_hidden_size * 6 + config.caps_out)
        self.dropout1 = nn.Dropout(config.dropout1)
        self.linear = nn.Linear(config.lstm_hidden_size * 2 + config.gru_hidden_size * 6 + config.caps_out,
                                config.out_size)
        self.dropout2 = nn.Dropout(config.dropout2)
        self.classifier = nn.Linear(config.out_size, config.num_labels)

    def forward(self, input_ids, labels):
        h_embedding = self.embedding(input_ids)
        h_embedding = self.apply_spatial_dropout(h_embedding)
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)
        content3 = self.capsule(h_gru)
        batch_size = content3.size(0)
        content3 = content3.view(batch_size, -1)
        content3 = self.dropout_caps(content3)
        content3 = F.relu(self.lin_caps(content3))
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        conc = torch.cat((h_lstm_atten, h_gru_atten, content3, avg_pool, max_pool), 1)
        conc = self.norm(conc)
        conc = self.dropout1(conc)
        conc = F.relu(conc)
        conc = self.linear(conc)
        conc = self.dropout2(conc)
        logits = self.classifier(conc)
        outputs = (logits,)
        if labels is not None:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
        return outputs
