import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from ..bases import TrainModel

class DPCNN(TrainModel):
    """
    Reference:
        Deep Pyramid Convolutional Neural Networks for Text Categorization
    """
    def __init__(self, config):
        super(DPCNN, self).__init__(config)
        self.num_kernels = config.num_kernels
        self.pooling_stride = config.pooling_stride
        self.kernel_size = config.kernel_size
        self.radius = int(self.kernel_size / 2)
        assert self.kernel_size % 2 == 1, "DPCNN kernel should be odd!"
        self.convert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(config.embedding_dim, self.num_kernels,self.kernel_size, padding=self.radius)
        )
        self.convs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius)
        ) for _ in range(config.DPCNN.blocks + 1)])

        self.linear = torch.nn.Linear(self.num_kernels, config.num_labels)


    def forward(self, input_ids,labels):
        embedding = self.token_embedding(input_ids)
        embedding = embedding.permute(0, 2, 1)
        conv_embedding = self.convert_conv(embedding)
        conv_features = self.convs[0](conv_embedding)
        conv_features = conv_embedding + conv_features
        for i in range(1, len(self.convs)):
            block_features = F.max_pool1d(
                conv_features, self.kernel_size, self.pooling_stride)
            conv_features = self.convs[i](block_features)
            conv_features = conv_features + block_features
        doc_embedding = F.max_pool1d(
            conv_features, conv_features.size(2)).squeeze()
        logits = self.dropout(self.linear(doc_embedding))
        outputs = (logits,)
        if labels is not None:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
        return outputs