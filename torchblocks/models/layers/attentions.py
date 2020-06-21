import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,d_model,dropout):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                "The d_model (%d) is not a multiple of the number of attention "
                "heads (%d)" % (d_model, num_heads))

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.d_k)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,mask = None):

        q_layer = self.query(q)
        k_layer = self.key(k)
        v_layer = self.value(v)

        query_layer = self.transpose_for_scores(q_layer)
        key_layer = self.transpose_for_scores(k_layer)
        value_layer = self.transpose_for_scores(v_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if mask is not None:
            # mask 1：表示真实的 ，0时padding
            mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            mask = (1.0 - mask) * -10000.0
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer,)
        return outputs

class Attention(nn.Module):

    def __init__(self, feature_dim, maxlen=70):
        super().__init__()
        self.attention_fc = nn.Linear(feature_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1, maxlen, 1, requires_grad=True))

    def forward(self, rnn_output):
        """
        forward attention scores and attended vectors
        :param rnn_output: (#batch, #seq_len, #feature)
        :return: attended_outputs (#batch, #feature)
        """
        attention_weights = self.attention_fc(rnn_output)
        seq_len = rnn_output.size(1)
        attention_weights = self.bias[:, :seq_len, :] + attention_weights
        attention_weights = torch.tanh(attention_weights)
        attention_weights = torch.exp(attention_weights)
        attention_weights_sum = torch.sum(attention_weights, dim=1, keepdim=True) + 1e-7
        attention_weights = attention_weights / attention_weights_sum
        attended = torch.sum(attention_weights * rnn_output, dim=1)
        return attended

class CosAttention(nn.Module):
    def __init__(self):
        super(CosAttention,self).__init__()

    def forward(self, title_output, attr_output):
        '''
        title_output (batchsize, seqlen, hidden_dim)
        attr_output (batchsize, hidden_dim)
        '''
        seq_len = title_output.size()[1]
        attr_output = attr_output.unsqueeze(1).repeat(1,seq_len,1)
        cos_sim = torch.cosine_similarity(attr_output,title_output,-1)
        cos_sim = cos_sim.unsqueeze(-1)
        outputs = title_output*cos_sim
        return outputs