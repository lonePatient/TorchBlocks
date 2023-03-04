import torch
import copy
import torch.nn as nn
import torch.nn.functional as F


class ResidualLSTM(nn.Module):
    def __init__(self, d_model, rnn='GRU', rnn_dropout_rate=0.2, dropout_rate=0.2):
        super(ResidualLSTM, self).__init__()
        self.downsample = nn.Linear(d_model, d_model // 2)
        if rnn == 'GRU':
            self.LSTM = nn.GRU(d_model // 2, d_model // 2, num_layers=2, bidirectional=False, dropout=rnn_dropout_rate)
        else:
            self.LSTM = nn.LSTM(d_model // 2, d_model // 2, num_layers=2, bidirectional=False, dropout=rnn_dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.linear1 = nn.Linear(d_model // 2, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = copy.deepcopy(x)
        x = self.downsample(x)
        x, _ = self.LSTM(x)
        x = self.dropout1(x)
        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout2(x)
        x = res + x
        return self.norm2(x)

class LSTMCharEncoder(nn.Module):
    r"""
    char级别基于LSTM的encoder.
    """

    def __init__(self, char_emb_size=50, hidden_size=None):
        r"""
        :param int char_emb_size: char级别embedding的维度. Default: 50
                例: 有26个字符, 每一个的embedding是一个50维的向量, 所以输入的向量维度为50.
        :param int hidden_size: LSTM隐层的大小, 默认为char的embedding维度
        :param initial_method: 初始化参数的方式, 默认为`xavier normal`
        """
        super(LSTMCharEncoder, self).__init__()
        self.hidden_size = char_emb_size if hidden_size is None else hidden_size
        self.lstm = nn.LSTM(input_size=char_emb_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)

    def forward(self, x):
        r"""
        :param torch.Tensor x: ``[ n_batch*n_word, word_length, char_emb_size]`` 输入字符的embedding
        :return: torch.Tensor : [ n_batch*n_word, char_emb_size]经过LSTM编码的结果
        """
        batch_size = x.shape[0]
        h0 = torch.empty(1, batch_size, self.hidden_size)
        h0 = nn.init.orthogonal_(h0)
        c0 = torch.empty(1, batch_size, self.hidden_size)
        c0 = nn.init.orthogonal_(c0)

        _, hidden = self.lstm(x, (h0, c0))
        return hidden[0].squeeze().unsqueeze(2)