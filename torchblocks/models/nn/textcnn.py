import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from ..bases.model_base import TrainModel

class TextCNN(TrainModel):
    def __init__(self, config):
        super(TextCNN, self).__init__(config)

        self.kernel_sizes = config.kernel_sizes
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(config.embedding_size,
                                              config.num_kernels,
                                              kernel_size,
                                              padding=kernel_size - 1))

        self.top_k = config.top_k_max_pooling
        hidden_size = len(self.kernel_sizes) * config.num_kernels * self.top_k
        self.linear = torch.nn.Linear(hidden_size, config.num_labels)
        self.dropout = torch.nn.Dropout(p=config.hidden_layer_dropout)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.init_weights()

    def forward(self, input_ids,labels=None):
        embedding = self.token_embedding(input_ids)
        embedding = embedding.transpose(1, 2)
        pooled_outputs = []
        for i, conv in enumerate(self.convs):
            convolution = F.relu(conv(embedding))
            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)
            pooled_outputs.append(pooled)
        doc_embedding = torch.cat(pooled_outputs, 1)
        logits = self.dropout(self.linear(doc_embedding))
        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
